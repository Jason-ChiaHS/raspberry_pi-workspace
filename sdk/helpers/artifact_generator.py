import queue
import threading
import typing as t
from pathlib import Path
from sqlite3 import Cursor
from .config import Config
import time

from ..sqlite_adapter import connect_to_sqlite
from .logger import logger


class ArtifactGenerator:
    def __init__(self, artifact_folder_path: Path):
        self.artifact_folder_path = artifact_folder_path
        self.artifact_folder_path.mkdir(parents=True, exist_ok=True)


class ProfilerArtifactGenerator(ArtifactGenerator):
    def __init__(self, artifact_folder_path: Path, db_file_name: str):
        super().__init__(artifact_folder_path)
        self.db_file_name = db_file_name
        self.jobs = queue.Queue()
        self.thread = threading.Thread(target=self.handle_jobs)
        self.thread.start()

    def handle_jobs(self):
        with connect_to_sqlite(
            (self.artifact_folder_path / f"{self.db_file_name}.db")
        ) as con:
            cur = con.cursor()
            self.create_table(cur)
            con.commit()
            while (job := self.jobs.get()) is not None:
                insert_statement, data = job[0], job[1]
                cur.executemany(insert_statement, data)
                con.commit()

    def create_table(self, cur: Cursor):
        """
        OVERRIDE\n
        Should create the table/tables to then write to
        """
        pass

    def queue_insert_statement_with_data(self, insert_statement: str, data: t.List):
        self.jobs.put((insert_statement, data))

class CustomProfiler:
    """
    To allow for easier saving of data related to time taken to run code
    Its on the developer to configure it, but it should generally be used as it
    The __init__ of the class takes care of creating the db for the first time and not overwriting it

    This will probably add latency to any nested profiling, especially if saving it to a db is turned on
    But the profiling of the affected function should be accurate
    Use with care
    """

    def __init__(self, config: t.Optional[Config] = None, max_buffer: int = 50, run: bool = True):
        """
        Will only run if the config.profiling is true and run is true

        max_buffer: Max buffer count in memory for the profiler
        run: Controls if the profilier will run
        """
        if config is None:
            self.save_to_db = False
        else:
            self.save_to_db = config.generate_artifacts.profiling
        if self.save_to_db:
            self.artifact_folder_path = Path(config.generate_artifacts.artifact_directory)
            self.artifact_folder_path.mkdir(parents=True, exist_ok=True)
            self.db_file_name = "custom_profiling"
            self.db_file = self.artifact_folder_path / f"{self.db_file_name}.db"

            if not Path(self.db_file).exists():
                with connect_to_sqlite(self.db_file) as con:
                    cur = con.cursor()
                    cur.execute(
                        "CREATE TABLE custom_function_profiling(function_name TEXT NOT NULL, time FLOAT NOT NULL)"
                    )
                    con.commit()

        self.run = True if config is None else config.profiling and run
        self.max_buffer = max_buffer
        self.profiles: list[tuple[str, float]] = []
        self.running_profiles: dict[str, float] = {}

    def start_profile_function(self, function_name: str):
        if not self.run:
            return
        self.running_profiles[function_name] = time.perf_counter()

    def end_profile_function(self, function_name: str):
        if not self.run:
            return
        if function_name not in self.running_profiles:
            logger.warning(f"profiling {function_name} failed")
            return
        t = time.perf_counter() - self.running_profiles.pop(function_name)
        if len(self.profiles) >= self.max_buffer:
            self.profiles.pop(0)
        self.profiles.append((function_name, t))
        if self.save_to_db:
            with connect_to_sqlite(self.db_file) as con:
                cur = con.cursor()
                cur.execute(
                    "INSERT INTO custom_function_profiling(function_name, time) VALUES (?, ?)",
                    [function_name, t],
                )
