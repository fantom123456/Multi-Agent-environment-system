print("LOADED policy.py")
import math
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Tuple


def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def mat_vec(M: List[List[float]], v: List[float]) -> List[float]:
    return [dot(row, v) for row in M]


def outer(v: List[float]) -> List[List[float]]:
    return [[vi * vj for vj in v] for vi in v]


def mat_add(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]


def vec_add(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def identity(n: int) -> List[List[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def invert_matrix(A: List[List[float]]) -> List[List[float]]:
    """
    Small dense Gauss-Jordan inverse. Fine for low-d feature vectors.
    """
    n = len(A)
    # augmented [A | I]
    aug = [A[i][:] + identity(n)[i][:] for i in range(n)]

    for col in range(n):
        # pivot
        pivot = col
        for r in range(col, n):
            if abs(aug[r][col]) > abs(aug[pivot][col]):
                pivot = r
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("Singular matrix")

        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]

        # normalize pivot row
        pv = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= pv

        # eliminate
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if abs(factor) < 1e-12:
                continue
            for j in range(2 * n):
                aug[r][j] -= factor * aug[col][j]

    inv = [row[n:] for row in aug]
    return inv


@dataclass
class LinUCBArm:
    d: int
    lam: float = 1.0
    A: List[List[float]] = None
    b: List[float] = None

    def __post_init__(self):
        if self.A is None:
            self.A = identity(self.d)
            # scale by lambda
            for i in range(self.d):
                self.A[i][i] *= self.lam
        if self.b is None:
            self.b = [0.0 for _ in range(self.d)]

    def score(self, x: List[float], alpha: float) -> float:
        A_inv = invert_matrix(self.A)
        theta = mat_vec(A_inv, self.b)
        mu = dot(theta, x)
        # uncertainty term sqrt(x^T A^-1 x)
        Ax = mat_vec(A_inv, x)
        sigma = math.sqrt(max(1e-12, dot(x, Ax)))
        return mu + alpha * sigma
     
    def update(self, x: List[float], r: float):
        self.A = mat_add(self.A, outer(x))
        self.b = vec_add(self.b, [r * xi for xi in x])


class GlobalPolicy:
    """
    Global (shared) LinUCB for two independent decision points:
      - proactive_action in {noop, ping, broadcast}
      - reactive_action in {ignore, reply, collab}

    Persisted to SQLite so learning carries across runs.
    """

    def __init__(
        self,
        db_path: str = "policy.db",
        alpha: float = 1.5,
        lam: float = 1.0,
        d: int = 8,
    ):
        self.db_path = db_path
        self.alpha = alpha
        self.lam = lam
        self.d = d

        self.proactive_arms: Dict[str, LinUCBArm] = {k: LinUCBArm(d=d, lam=lam) for k in ["noop", "ping", "broadcast"]}
        self.reactive_arms: Dict[str, LinUCBArm] = {k: LinUCBArm(d=d, lam=lam) for k in ["ignore", "reply", "collab"]}

        self._load()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self, con: sqlite3.Connection):
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS linucb_arms (
              decision TEXT NOT NULL,
              arm TEXT NOT NULL,
              d INTEGER NOT NULL,
              A_json TEXT NOT NULL,
              b_json TEXT NOT NULL,
              PRIMARY KEY(decision, arm)
            );
            """
        )
        con.commit()

    def _load(self):
        con = self._connect()
        self._init_db(con)
        cur = con.cursor()

        def load_decision(decision: str, arms: Dict[str, LinUCBArm]):
            for arm_name, arm in arms.items():
                row = cur.execute(
                    "SELECT A_json, b_json, d FROM linucb_arms WHERE decision=? AND arm=?",
                    (decision, arm_name),
                ).fetchone()
                if not row:
                    continue
                A_json, b_json, d = row
                if int(d) != self.d:
                    # dimension mismatch; ignore old params
                    continue
                import json
                arm.A = json.loads(A_json)
                arm.b = json.loads(b_json)

        load_decision("proactive", self.proactive_arms)
        load_decision("reactive", self.reactive_arms)
        con.close()

    def save(self):
        con = self._connect()
        self._init_db(con)
        cur = con.cursor()
        import json

        def upsert(decision: str, arms: Dict[str, LinUCBArm]):
            for arm_name, arm in arms.items():
                cur.execute(
                    """
                    INSERT INTO linucb_arms(decision, arm, d, A_json, b_json)
                    VALUES(?, ?, ?, ?, ?)
                    ON CONFLICT(decision, arm) DO UPDATE SET
                      d=excluded.d, A_json=excluded.A_json, b_json=excluded.b_json
                    """,
                    (decision, arm_name, self.d, json.dumps(arm.A), json.dumps(arm.b)),
                )

        upsert("proactive", self.proactive_arms)
        upsert("reactive", self.reactive_arms)
        con.commit()
        con.close()

    # -------- feature extraction --------
    def featurize(
        self,
        *,
        inbox_count: int,
        sent_count: int,
        known_agents: int,
        seconds_since_proactive: float,
        last_msg_kind: str | None,
    ) -> List[float]:
        """
        Small, stable feature vector (d=8 by default).
        Keep it simple; you can add more later.
        """
        # normalize-ish
        ic = math.log1p(inbox_count)
        sc = math.log1p(sent_count)
        ka = math.log1p(known_agents)
        sp = min(5.0, seconds_since_proactive) / 5.0

        kind_is_ping = 1.0 if last_msg_kind == "ping" else 0.0
        kind_is_broadcast = 1.0 if last_msg_kind == "broadcast" else 0.0
        kind_is_reply = 1.0 if last_msg_kind == "reply" else 0.0

        bias = 1.0
        return [bias, ic, sc, ka, sp, kind_is_ping, kind_is_broadcast, kind_is_reply]

    # -------- decisions --------
    def choose_proactive(self, x: List[float]) -> str:
        scored: List[Tuple[str, float]] = []
        for a, arm in self.proactive_arms.items():
            scored.append((a, arm.score(x, self.alpha)))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[0][0]

    def choose_reactive(self, x: List[float]) -> str:
        scored: List[Tuple[str, float]] = []
        for a, arm in self.reactive_arms.items():
            scored.append((a, arm.score(x, self.alpha)))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[0][0]

    # -------- learning --------
    def update_proactive(self, arm: str, x: List[float], reward: float):
        
        self.proactive_arms[arm].update(x, reward)

    def update_reactive(self, arm: str, x: List[float], reward: float):
        
        self.reactive_arms[arm].update(x, reward)