import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from sqlalchemy import create_engine, Column, String, Text, DateTime, JSON, inspect
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class DigitizationRun(Base):
    __tablename__ = "medidigitizer_runs"

    thread_id = Column(String(50), primary_key=True)
    image_name = Column(String(255), nullable=False)
    image_path = Column(Text, nullable=False)
    merged_ocr_text = Column(Text, default="")
    ner_data = Column(JSON, default=dict)
    validation_warnings = Column(JSON, default=list)
    chat_history = Column(JSON, default=list)
    status = Column(String(50), default="pending")  # pending, approved
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "image_name": self.image_name,
            "image_path": self.image_path,
            "merged_ocr_text": self.merged_ocr_text,
            "ner_data": self.ner_data,
            "validation_warnings": self.validation_warnings,
            "chat_history": self.chat_history or [],
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

class DatabaseHistoryManager:
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self.backend = "sqlite"
        self._init_db()

    def _init_db(self):
        # Read from environment variables if set, otherwise default
        pg_user = os.getenv("DB_USER", "postgres")
        pg_password = os.getenv("DB_PASSWORD", "password")
        pg_host = os.getenv("DB_HOST", "localhost")
        pg_port = os.getenv("DB_PORT", "5432")
        pg_db = os.getenv("DB_NAME", "medidigitizer_history")

        # 1. Try connecting to PostgreSQL
        pg_url = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"
        try:
            # Short timeout to fail fast if PG is offline
            self.engine = create_engine(pg_url, connect_args={"connect_timeout": 3})
            # Test connection
            with self.engine.connect() as conn:
                pass
            self.backend = "postgresql"
            print(f"[Database History] Connected successfully to PostgreSQL: {pg_host}:{pg_port}/{pg_db}", flush=True)
        except Exception as pg_err:
            print(f"[Database History] PostgreSQL connection failed: {pg_err}. Trying to auto-create database...", flush=True)
            # Try to auto-create PG database if server is running but database is missing
            try:
                temp_url = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/postgres"
                temp_engine = create_engine(temp_url, connect_args={"connect_timeout": 2})
                with temp_engine.connect() as temp_conn:
                    # Execute database creation in autocommit
                    temp_conn.execution_options(isolation_level="AUTOCOMMIT").execute(f'CREATE DATABASE "{pg_db}"')
                self.engine = create_engine(pg_url)
                self.backend = "postgresql"
                print(f"[Database History] Auto-created and connected to PostgreSQL database: {pg_db}", flush=True)
            except Exception as create_err:
                print(f"[Database History] Auto-creation failed: {create_err}. Falling back to SQLite.", flush=True)
                # Fallback to local SQLite db file
                db_path = Path(__file__).resolve().parent.parent / "history.db"
                sqlite_url = f"sqlite:///{db_path}"
                self.engine = create_engine(sqlite_url)
                self.backend = "sqlite"
                print(f"[Database History] Initialized local SQLite fallback at: {db_path}", flush=True)

        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(bind=self.engine)

        # Auto-migrate: check if chat_history column exists, if not add it
        try:
            inspector = inspect(self.engine)
            columns = [c["name"] for c in inspector.get_columns("medidigitizer_runs")]
            if "chat_history" not in columns:
                print("[Database History] Running migration: adding chat_history column...", flush=True)
                col_type = "JSON" if self.backend == "postgresql" else "TEXT"
                with self.engine.connect() as conn:
                    # Execute migration and commit
                    conn.execute(self.engine.dialect.ddl_compiler(self.engine.dialect, None).construct_compiler().connection.execute(f"ALTER TABLE medidigitizer_runs ADD COLUMN chat_history {col_type}"))
                    if self.backend == "sqlite":
                        conn.execute("commit")
                print("[Database History] Migration completed successfully.", flush=True)
        except Exception as mig_err:
            # Simple direct alter table execution in case of dialect compile warning
            try:
                with self.engine.connect() as conn:
                    col_type = "JSON" if self.backend == "postgresql" else "TEXT"
                    # SQLAlchemy 2.0 connection execute requires text or connection isolation
                    from sqlalchemy import text
                    conn.execute(text(f"ALTER TABLE medidigitizer_runs ADD COLUMN chat_history {col_type}"))
                    conn.commit()
                print("[Database History] Simple migration completed successfully.", flush=True)
            except Exception as simple_err:
                print(f"[Database History Migration Warning]: {simple_err}", flush=True)

        # Ensure default historical run (fe6126fa) is seeded
        self._seed_default_run()

    def _seed_default_run(self):
        try:
            db = self.SessionLocal()
            try:
                run = db.query(DigitizationRun).filter(DigitizationRun.thread_id == "fe6126fa").first()
                if not run:
                    seed_path = Path(__file__).resolve().parent / "seed_data.json"
                    if seed_path.exists():
                        with open(seed_path, "r", encoding="utf-8") as f:
                            seed = json.load(f)
                        created_dt = None
                        if seed.get("created_at"):
                            try:
                                created_dt = datetime.fromisoformat(seed["created_at"])
                            except Exception:
                                created_dt = datetime.utcnow()
                        run = DigitizationRun(
                            thread_id=seed.get("thread_id", "fe6126fa"),
                            image_name=seed.get("image_name", "sample1.jpg"),
                            image_path=seed.get("image_path", ""),
                            merged_ocr_text=seed.get("merged_ocr_text", ""),
                            ner_data=seed.get("ner_data", {}),
                            validation_warnings=seed.get("validation_warnings", []),
                            chat_history=seed.get("chat_history", []),
                            status=seed.get("status", "approved"),
                            created_at=created_dt or datetime.utcnow()
                        )
                        db.add(run)
                        db.commit()
                        print("[Database History] Seeded default approved run 'fe6126fa' successfully.", flush=True)
            finally:
                db.close()
        except Exception as seed_err:
            print(f"[Database History Seed Warning]: {seed_err}", flush=True)

    def save_run(
        self,
        thread_id: str,
        image_name: str,
        image_path: str,
        merged_ocr_text: str = "",
        ner_data: Optional[Dict[str, Any]] = None,
        validation_warnings: Optional[List[str]] = None,
        status: str = "pending"
    ) -> Dict[str, Any]:
        """Saves a run log or updates it if thread_id already exists."""
        db = self.SessionLocal()
        try:
            run = db.query(DigitizationRun).filter(DigitizationRun.thread_id == thread_id).first()
            if not run:
                run = DigitizationRun(
                    thread_id=thread_id,
                    image_name=image_name,
                    image_path=image_path,
                    merged_ocr_text=merged_ocr_text,
                    ner_data=ner_data or {},
                    validation_warnings=validation_warnings or [],
                    status=status
                )
                db.add(run)
            else:
                run.image_name = image_name
                run.image_path = image_path
                run.merged_ocr_text = merged_ocr_text
                run.ner_data = ner_data or run.ner_data
                run.validation_warnings = validation_warnings or run.validation_warnings
                run.status = status
                run.updated_at = datetime.utcnow()
            
            db.commit()
            db.refresh(run)
            return run.to_dict()
        finally:
            db.close()

    def get_run(self, thread_id: str) -> Optional[Dict[str, Any]]:
        db = self.SessionLocal()
        try:
            run = db.query(DigitizationRun).filter(DigitizationRun.thread_id == thread_id).first()
            return run.to_dict() if run else None
        finally:
            db.close()

    def list_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        db = self.SessionLocal()
        try:
            runs = db.query(DigitizationRun).order_by(DigitizationRun.created_at.desc()).limit(limit).all()
            return [r.to_dict() for r in runs]
        finally:
            db.close()

    def delete_run(self, thread_id: str) -> bool:
        db = self.SessionLocal()
        try:
            run = db.query(DigitizationRun).filter(DigitizationRun.thread_id == thread_id).first()
            if run:
                db.delete(run)
                db.commit()
                return True
            return False
        finally:
            db.close()

    def save_chat_message(self, thread_id: str, role: str, content: str) -> List[Dict[str, str]]:
        """Appends a user or assistant chat message to a run log's history list."""
        db = self.SessionLocal()
        try:
            run = db.query(DigitizationRun).filter(DigitizationRun.thread_id == thread_id).first()
            if run:
                history = list(run.chat_history or [])
                history.append({"role": role, "content": content})
                run.chat_history = history
                run.updated_at = datetime.utcnow()
                db.commit()
                return history
            return []
        finally:
            db.close()
