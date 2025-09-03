"""
Inspired by https://github.com/Marcus-Peterson/tursopy

Forking and placing it directly here as I need more flexibility in the statments
"""

import requests

class TursoConnection:
    def __init__(self, database_url, auth_token=None):
        self.database_url = database_url
        self.auth_token = auth_token

        self.headers = {
            'Content-Type': 'application/json'
        }
        if self.auth_token is not None:
            self.headers["Authorization"] = f"Bearer {self.auth_token}"


    def execute_query(self, sql, args=None):
        """Execute a single SQL statement with optional arguments."""
        payload = {
            'requests': [
                {
                    'type': 'execute',
                    'stmt': {
                        'sql': sql,
                        'args': args or []
                    }
                },
                {'type': 'close'}
            ]
        }
        response = requests.post(f'{self.database_url}/v2/pipeline', json=payload, headers=self.headers)
        return self._handle_response(response)

    def execute_pipeline(self, queries):
        """Execute a series of SQL statements."""
        payload = {'requests': queries + [{'type': 'close'}]}
        response = requests.post(f'{self.database_url}/v2/pipeline', json=payload, headers=self.headers)
        return self._handle_response(response)

    @staticmethod
    def _handle_response(response):
        if response.status_code == 200:
            return response.json()
        else:
            raise RuntimeError(f"Turso API Error {response.status_code}: {response.text}")
