import math
import numpy as np
from ultralytics import YOLO

class ChessDetector:
    def __init__(self, weights_path, conf=0.4):
        self.model = YOLO(weights_path)
        self.conf = conf
        self.pieces = {name: [] for name in [
            'bRook','bKnight','bBishop','bQueen','bKing','bPawn',
            'wRook','wKnight','wBishop','wQueen','wKing','wPawn'
        ]}
        self.corners = []
        self.board_matrix = {}
        self.square_map = {}
        
    ############################################################################
    
    def resolve_queen_king_conflict(self, 
                                    pieces: dict, 
                                    iou_threshold: float = 0.5, 
                                    conf_gap: float = 0.2):
        """
        Fix cases where the model outputs two wQueens, one of which is actually a wKing.
        Keeps the higher-confidence queen, reassigns the weaker one as king if confidence gap is large.
        """
        def iou(det1, det2):
            x1, y1 = det1["cntr"][0] - det1["width"]/2, det1["cntr"][1] - det1["height"]/2
            x2, y2 = det1["cntr"][0] + det1["width"]/2, det1["cntr"][1] + det1["height"]/2
            xx1, yy1 = det2["cntr"][0] - det2["width"]/2, det2["cntr"][1] - det2["height"]/2
            xx2, yy2 = det2["cntr"][0] + det2["width"]/2, det2["cntr"][1] + det2["height"]/2

            inter_x1, inter_y1 = max(x1, xx1), max(y1, yy1)
            inter_x2, inter_y2 = min(x2, xx2), min(y2, yy2)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

            area1 = (x2 - x1) * (y2 - y1)
            area2 = (xx2 - xx1) * (yy2 - yy1)
            union = area1 + area2 - inter_area
            return inter_area / union if union > 0 else 0

        queens = pieces.get("wQueen", [])
        kings = pieces.get("wKing", [])

        cleaned_queens = []
        for i, q in enumerate(queens):
            is_duplicate = False
            for j, other in enumerate(queens):
                if i == j:
                    continue
                if iou(q, other) > iou_threshold:
                    # If they overlap and confidence gap is big
                    if abs(q["conf"] - other["conf"]) > conf_gap:
                        # Higher confidence stays queen
                        if q["conf"] > other["conf"]:
                            cleaned_queens.append(q)
                            # Reassign the weaker one as king
                            kings.append(other)
                        else:
                            cleaned_queens.append(other)
                            kings.append(q)
                        is_duplicate = True
                        break
            if not is_duplicate:
                cleaned_queens.append(q)

        pieces["wQueen"] = cleaned_queens
        pieces["wKing"] = kings
        return pieces

    def resolve_black_queen_king_conflict(self, 
                                          pieces: dict, 
                                          iou_threshold: float = 0.5, 
                                          conf_gap: float = 0.25):
        """
        Fix cases where the model outputs two bKings, one of which is actually a bQueen.
        Keeps the higher-confidence king, reassigns the weaker one as queen if confidence gap is large.
        """
        def iou(det1, det2):
            x1, y1 = det1["cntr"][0] - det1["width"]/2, det1["cntr"][1] - det1["height"]/2
            x2, y2 = det1["cntr"][0] + det1["width"]/2, det1["cntr"][1] + det1["height"]/2
            xx1, yy1 = det2["cntr"][0] - det2["width"]/2, det2["cntr"][1] - det2["height"]/2
            xx2, yy2 = det2["cntr"][0] + det2["width"]/2, det2["cntr"][1] + det2["height"]/2

            inter_x1, inter_y1 = max(x1, xx1), max(y1, yy1)
            inter_x2, inter_y2 = min(x2, xx2), min(y2, yy2)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

            area1 = (x2 - x1) * (y2 - y1)
            area2 = (xx2 - xx1) * (yy2 - yy1)
            union = area1 + area2 - inter_area
            return inter_area / union if union > 0 else 0

        kings = pieces.get("bKing", [])
        queens = pieces.get("bQueen", [])

        cleaned_kings = []
        for i, k in enumerate(kings):
            is_duplicate = False
            for j, other in enumerate(kings):
                if i == j:
                    continue
                if iou(k, other) > iou_threshold:
                    # If they overlap and confidence gap is big
                    if abs(k["conf"] - other["conf"]) > conf_gap:
                        # Higher confidence stays king
                        if k["conf"] > other["conf"]:
                            cleaned_kings.append(k)
                            # Reassign the weaker one as queen
                            queens.append(other)
                        else:
                            cleaned_kings.append(other)
                            queens.append(k)
                        is_duplicate = True
                        break
            if not is_duplicate:
                cleaned_kings.append(k)

        pieces["bKing"] = cleaned_kings
        pieces["bQueen"] = queens
        return pieces

    def build_board_with_pieces(self, pieces: dict, corners: list, side: str = 'b'):
        """
        Create the board matrix and square map, then assign detected pieces to squares.

        :param pieces: dict of detected pieces, e.g. {"wPawn": [{"cntr": (x,y), "conf": 0.9}, ...]}
        :param corners: list of 4 corner points of the board [(x,y), ...]
        :param side: 'b' if camera is on black's side, 'w' if on white's side
        :return: (board_matrix, square_map)
        """
        if len(corners) != 4:
            print("Need 4 corners to assign pieces to squares.")
            return None, None

        # --- Create board mapping ---
        if side == 'b':
            files = 'hgfedcba'
            ranks = range(1, 9)
        else:
            files = 'abcdefgh'
            ranks = range(8, 0, -1)

        board_matrix = {}
        square_map = {}
        for row, rank in enumerate(ranks):
            for col, file in enumerate(files):
                square = f"{file}{rank}"
                board_matrix[square] = None
                square_map[(row, col)] = square

        # --- Sort corners using the staticmethod ---
        top_left, top_right, bottom_left, bottom_right = self.sort_corners(corners)

        # --- Compute square boundaries ---
        left_edge = np.linspace(top_left, bottom_left, 9)
        right_edge = np.linspace(top_right, bottom_right, 9)

        square_bounds = {}
        for row in range(8):
            row_points_top = np.linspace(left_edge[row], right_edge[row], 9)
            row_points_bottom = np.linspace(left_edge[row+1], right_edge[row+1], 9)
            for col in range(8):
                p_tl = row_points_top[col]
                p_tr = row_points_top[col+1]
                p_bl = row_points_bottom[col]
                p_br = row_points_bottom[col+1]
                square_bounds[(row, col)] = (p_tl, p_tr, p_bl, p_br)

        # --- Assign pieces ---
        for cls_name, detections in pieces.items():
            for det in detections:
                cx, cy = det["cntr"]
                conf = det["conf"]
                for (row, col), (p_tl, p_tr, p_bl, p_br) in square_bounds.items():
                    min_x = min(p_tl[0], p_bl[0])
                    max_x = max(p_tr[0], p_br[0])
                    min_y = min(p_tl[1], p_tr[1])
                    max_y = max(p_bl[1], p_br[1])
                    if min_x <= cx <= max_x and min_y <= cy <= max_y:
                        square_name = square_map[(row, col)]
                        current = board_matrix[square_name]
                        if current is None or conf > current["conf"]:
                            board_matrix[square_name] = {
                                "piece": cls_name,
                                "conf": conf
                            }
                # no break → check all squares

        # Save for later use
        self.board_matrix = board_matrix
        self.square_map = square_map

        return board_matrix, square_map

    def board_to_fen(self, board_matrix: dict, side='b'):
        if board_matrix != None:
            piece_map = {
                'wPawn': 'P','wKnight': 'N','wBishop': 'B',
                'wRook': 'R','wQueen': 'Q','wKing': 'K',
                'bPawn': 'p','bKnight': 'n','bBishop': 'b',
                'bRook': 'r','bQueen': 'q','bKing': 'k'
            }
            fen_rows = []
            for rank in range(8, 0, -1):
                row_str = ""
                empty_count = 0
                for file in "abcdefgh":
                    square = f"{file}{rank}"
                    piece_entry = board_matrix.get(square)
                    piece_symbol = None
                    if isinstance(piece_entry, dict):
                        piece_symbol = piece_map.get(piece_entry["piece"])
                    elif isinstance(piece_entry, str):
                        piece_symbol = piece_map.get(piece_entry)
                    if piece_symbol:
                        if empty_count > 0:
                            row_str += str(empty_count)
                            empty_count = 0
                        row_str += piece_symbol
                    else:
                        empty_count += 1
                if empty_count > 0:
                    row_str += str(empty_count)
                fen_rows.append(row_str)
            fen_position = "/".join(fen_rows)

            castling = ""
            if self.board_matrix.get("e1") in ("wKing", {"piece": "wKing"}):
                if self.board_matrix.get("h1") in ("wRook", {"piece": "wRook"}):
                    castling += "K"
                if self.board_matrix.get("a1") in ("wRook", {"piece": "wRook"}):
                    castling += "Q"
            if self.board_matrix.get("e8") in ("bKing", {"piece": "bKing"}):
                if self.board_matrix.get("h8") in ("bRook", {"piece": "bRook"}):
                    castling += "k"
                if self.board_matrix.get("a8") in ("bRook", {"piece": "bRook"}):
                    castling += "q"
            if castling == "":
                castling = "-"
            en_passant = "-"
            halfmove = 0
            fullmove = 1
            return f"{fen_position} {side} {castling} {en_passant} {halfmove} {fullmove}"

    @staticmethod
    def sort_corners(points):
        # points = [(x,y,conf), ...] or (x,y)
        pts = [(p[0], p[1]) for p in points]
        pts = sorted(pts, key=lambda p: p[1])  # sort by y
        top = sorted(pts[:2], key=lambda p: p[0])
        bottom = sorted(pts[2:], key=lambda p: p[0])
        return [top[0], top[1], bottom[0], bottom[1]]

    def predict(self, image_path, save=True, print_result=False):
        self.corners = []
        self.pieces = {name: [] for name in [
            'bRook','bKnight','bBishop','bQueen','bKing','bPawn',
            'wRook','wKnight','wBishop','wQueen','wKing','wPawn'
        ]}
        results = self.model.predict(source=image_path, save=save, conf=self.conf)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = self.model.names[cls_id]
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w / 2
                cy = y1 + h / 2
                detection = {
                    "conf": round(conf, 2),
                    "cntr": (int(cx), int(cy)),
                    "width": int(w),
                    "height": int(h)
                }
                if cls_name in self.pieces:
                    self.pieces[cls_name].append(detection)
                else:
                    self.corners.append((int(cx), int(cy), round(conf, 2)))
                if print_result:
                    print(f"{cls_name}\t- {conf:.2f}\t"
                          f"center=({int(cx)}, {int(cy)})\t"
                          f"w={int(w)}\th={int(h)}")
        return self.pieces, self.corners
        
    def summary(self):
        print("\n--- Detection Summary ---")
        if self.corners:
            print("Corners:", self.sort_corners(self.corners))
        for piece, detections in self.pieces.items():
            if detections:
                print(f"{piece}\t{len(detections)}\t{detections}")
