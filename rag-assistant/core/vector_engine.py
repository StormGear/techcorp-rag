"""
Vector Engine - Manages vector database operations using PostgreSQL + pgvector (Neon)
and OpenRouter (openai/text-embedding-3-small) for embeddings.
"""

import os
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
import psycopg2
from psycopg2.extras import Json, RealDictCursor
from pgvector.psycopg2 import register_vector
from openai import OpenAI
import dotenv

dotenv.load_dotenv()


class VectorEngine:
    def __init__(self, collection_name: str = "techcorp_chunks"):
        self.table_name = collection_name.replace("-", "_")
        self.db_url = os.environ.get("DATABASE_URL")
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")

        if not self.db_url:
            raise ValueError(
                "DATABASE_URL environment variable is not set. "
                "Please configure your Neon PostgreSQL connection string in .env"
            )

        if not self.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is not set. "
                "Please configure your OpenRouter API key in .env"
            )

        print("[VectorEngine] Initializing OpenRouter embedding client...")
        import httpx
        self.embedding_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.openrouter_api_key,
            http_client=httpx.Client()
        )
        self.embedding_model = "openai/text-embedding-3-small"
        self.embedding_dim = 1536

        print("[VectorEngine] Initializing PostgreSQL (Neon) with pgvector...")
        self._init_db()
        self.last_updated = datetime.now()

    def _get_connection(self):
        """Create a new PostgreSQL connection and register pgvector types"""
        conn = psycopg2.connect(self.db_url)
        register_vector(conn)
        return conn

    def _create_embedding(self, text: str) -> List[float]:
        """Generate embedding vector using OpenRouter API (text-embedding-3-small)"""
        clean_text = text.replace("\n", " ")
        response = self.embedding_client.embeddings.create(
            model=self.embedding_model,
            input=clean_text
        )
        return response.data[0].embedding

    def _init_db(self):
        """Ensure pgvector extension, table, and index exist in Neon with 1536 dimensions"""
        # Step 1: Ensure extension is enabled
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()

        # Step 2: Check for table dimension and create if needed
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Check column dimension if table already exists
                cur.execute("""
                    SELECT atttypmod 
                    FROM pg_attribute 
                    JOIN pg_class ON pg_class.oid = pg_attribute.attrelid
                    JOIN pg_namespace ON pg_namespace.oid = pg_class.relnamespace
                    WHERE pg_class.relname = %s AND attname = 'embedding';
                """, (self.table_name,))
                col = cur.fetchone()

                # If existing table was created with a different dimension (e.g. 384 from MiniLM), recreate it
                if col and col[0] != self.embedding_dim:
                    print(
                        f"[VectorEngine] Existing table has dimension {col[0]}, "
                        f"upgrading to {self.embedding_dim} for text-embedding-3-small..."
                    )
                    cur.execute(f"DROP TABLE IF EXISTS {self.table_name} CASCADE;")

                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id TEXT PRIMARY KEY,
                        text TEXT NOT NULL,
                        metadata JSONB,
                        embedding vector({self.embedding_dim})
                    );
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_hnsw_idx
                    ON {self.table_name} USING hnsw (embedding vector_cosine_ops);
                    """
                )
            conn.commit()

        print(f"[VectorEngine] Connected to Neon. Table '{self.table_name}' ({self.embedding_dim} dims) is ready.")

    def add_document(self, doc_id: str, text: str, metadata: Dict[str, Any]):
        """Add or update a document chunk in the vector store"""
        embedding = np.array(self._create_embedding(text), dtype=np.float32)

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self.table_name} (id, text, metadata, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE
                    SET text = EXCLUDED.text,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding;
                    """,
                    (doc_id, text, Json(metadata), embedding)
                )
            conn.commit()

        self.last_updated = datetime.now()

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents using cosine similarity"""
        query_embedding = np.array(self._create_embedding(query), dtype=np.float32)

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT
                        id,
                        text,
                        metadata,
                        1 - (embedding <=> %s) AS score
                    FROM {self.table_name}
                    ORDER BY embedding <=> %s
                    LIMIT %s;
                    """,
                    (query_embedding, query_embedding, limit)
                )
                rows = cur.fetchall()

        formatted_results = []
        for row in rows:
            meta = row['metadata'] if row['metadata'] is not None else {}
            score = float(row['score']) if row['score'] is not None else 0.0
            formatted_results.append({
                'id': row['id'],
                'text': row['text'],
                'metadata': meta,
                'score': score,
                'title': meta.get('title', 'Unknown'),
                'snippet': row['text'][:150] + '...'
            })

        return formatted_results

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        COUNT(*) AS total_chunks,
                        COUNT(DISTINCT metadata->>'file') AS total_documents
                    FROM {self.table_name};
                    """
                )
                row = cur.fetchone()
                total_chunks = row[0] if row else 0
                total_documents = row[1] if row else 0

        return {
            'total_chunks': total_chunks,
            'total_documents': total_documents,
            'last_updated': self.last_updated.isoformat()
        }

    def clear_collection(self):
        """Clear all documents from the collection"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {self.table_name};")
            conn.commit()
        self.last_updated = datetime.now()

    def is_initialized(self) -> bool:
        """Check if the vector store has been initialized with documents"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT EXISTS (SELECT 1 FROM {self.table_name} LIMIT 1);")
                res = cur.fetchone()
                return bool(res[0]) if res else False

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text (useful for visualization)"""
        return self._create_embedding(text)

    def get_visualization_data(self) -> Dict[str, Any]:
        """Get data for vector space visualization"""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"""
                        SELECT embedding, metadata
                        FROM {self.table_name}
                        LIMIT 50;
                        """
                    )
                    rows = cur.fetchall()

            if not rows:
                return {"error": "No documents in collection"}

            points = []
            for row in rows:
                embedding = row['embedding']
                meta = row['metadata'] or {}
                points.append({
                    'x': float(embedding[0]),
                    'y': float(embedding[1]),
                    'category': meta.get('category', 'unknown'),
                    'title': meta.get('title', 'Unknown')
                })

            return {
                'points': points,
                'categories': list(set(p['category'] for p in points))
            }
        except Exception as e:
            return {"error": str(e)}