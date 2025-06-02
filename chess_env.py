class ChessEnv(gym.Env):
    def __init__(self):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()
        self.action_space = spaces.Discrete(4672)
        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 8, 12), dtype=np.float32)

    def reset(self):
        self.board.reset()
        return self.board_to_tensor()

    def step(self, action_index):
        legal_moves = list(self.board.legal_moves)
        done = False
        reward = 0

        if len(legal_moves) == 0:
            return self.board_to_tensor(), -1, True, {}

        move = legal_moves[action_index % len(legal_moves)]
        self.board.push(move)

        # Simple reward
        if self.board.is_checkmate():
            reward = 1
            done = True
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            reward = 0.5
            done = True

        return self.board_to_tensor(), reward, done, {}

    def render(self, mode='human'):
        print(self.board)

    def board_to_tensor(self):
        tensor = np.zeros((8, 8, 12), dtype=np.float32)
        piece_map = self.board.piece_map()

        for square, piece in piece_map.items():
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            idx = self.piece_index(piece)
            tensor[row, col, idx] = 1

        return tensor

    def piece_index(self, piece):
        return {
            (chess.PAWN, chess.WHITE): 0,
            (chess.KNIGHT, chess.WHITE): 1,
            (chess.BISHOP, chess.WHITE): 2,
            (chess.ROOK, chess.WHITE): 3,
            (chess.QUEEN, chess.WHITE): 4,
            (chess.KING, chess.WHITE): 5,
            (chess.PAWN, chess.BLACK): 6,
            (chess.KNIGHT, chess.BLACK): 7,
            (chess.BISHOP, chess.BLACK): 8,
            (chess.ROOK, chess.BLACK): 9,
            (chess.QUEEN, chess.BLACK): 10,
            (chess.KING, chess.BLACK): 11,
        }[(piece.piece_type, piece.color)]
