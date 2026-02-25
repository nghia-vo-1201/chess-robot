import chess
import chess.engine
import paramiko
from scp import SCPClient, SCPException
import os
import cv2
import time
from detect_chess_board import ChessDetector

class ChessGame:
    def __init__(self, engine_path, 
                 ssh_host='192.168.200.1', 
                 ssh_port=22, ssh_user='root', 
                 ssh_pass='dobot'):
        # Initialize board and engine
        self.board = chess.Board()
        self.prevBoard = self.board
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.engine.configure({"UCI_LimitStrength": True,
                               "UCI_Elo": 1600,
                               "Skill Level": 10,
                               })

        # SSH connection to robot
        self.commandRobotDir = '/dobot/userdata/project/project/sshcom/moveRobot.txt'
        self.statusRobotDir = '/dobot/userdata/project/project/sshcom/statusRobot.txt'
        self.ssh = self.create_SSH_client(ssh_host, ssh_port, ssh_user, ssh_pass)

        # Game state variables
        self.game_start = False
        self.game_end = False
        self.prevBoard = ''
        self.sidePlayer = 'w'
        self.sideRobot = 'b'
        self.typeMove = ''
        self.moveRobot = ''
        self.winner = ''
        self.forfeitGame = False
        self.firstMove = False
        self.firstPrint = False
        self.commandSendDir = 'sendCommand.txt'
        self.statusGetDir = 'getStatus.txt'

        # Captured piece coordinates
        self.capturedAmount = 0
        self.initialX = -60
        self.initialY = 160
        self.capturedX = 0
        self.capturedY = 0
        self.coefX = -40        # Horizontal spacing between captured pieces
        self.coefY = -40        # Vertical spacing between captured pieces
        self.capturedBatch = 5  # Number of captured pieces on a row/column (Max 6!!)
        
        # Detector instances
        self.detector = ChessDetector("C:/Users/VLK_DEV/Desktop/NghiaVo/chess_robot/runs/detect/train/weights/best.pt")
        self.corners = []
        self.read_corner_once = False

        # --- Webcam opened once ---
        self.cap = cv2.VideoCapture(0)
        
        # Disable autofocus
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_FOCUS, 0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


        # Set manual focus (value range depends on driver, often 0–255)
        #self.cap.set(cv2.CAP_PROP_FOCUS, 30)
        
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open webcam.")

        # Try setting high resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        

        print("Webcam initialized and warmed up.")
        
        
    def launch_board_gui(self):
        """
        Open a Tkinter window that shows the chess board as a GUI.
        The board updates live whenever self.board changes.
        Buttons:
        - Green: start a new game (only works when paused)
        - Red: stop game
        - Switch Side: flip player colour (only works when paused)
        - End: completely stop the game loop
        Also shows two StringVars: one for game status, one for log.
        """
        import tkinter as tk
        import chess

        glyphs = {
            "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
            "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚"
        }

        SQUARE_SIZE = 64
        LIGHT = "#f8ccbc"
        DARK = "#87b886"

        root = tk.Tk()
        root.title("Chess Robot")
        icon = tk.PhotoImage(file="icon/ico/favicon-64x64.ico")
        large_icon = icon.zoom(2, 2)
        root.iconphoto(True, icon)   # True = apply to all windows

        canvas = tk.Canvas(root, width=8*SQUARE_SIZE, height=8*SQUARE_SIZE)
        canvas.pack()

        # Track orientation: "b" = white at bottom, "w" = black at bottom
        orientation = tk.StringVar(value="b")

        # --- NEW StringVars for status and log ---
        self.status_var = tk.StringVar(value='Press "Start New" to start the a new game!')  # Game status
        self.log_var = tk.StringVar(value='Press "Help" for instruction in detail')     # Log messages

        # Labels to display them
        status_label = tk.Label(root, textvariable=self.status_var,
                                font=("Arial", 12), fg="#635dfc")
        status_label.pack(pady=5)

        log_label = tk.Label(root, textvariable=self.log_var,
                            font=("Arial", 10), fg="gray")
        log_label.pack(pady=5)

        # --- Button actions ---
        def on_green():
            if not self.game_start:
                import shutil
                folder_to_delete = "captured_images"
                if os.path.exists(folder_to_delete):
                    try:
                        shutil.rmtree(folder_to_delete)
                    except Exception as e:
                        self.log_var.set(f"Error deleting folder: {e}")
                self.board = chess.Board()
                self.game_start = True
                self.firstMove = False
                self.capturedAmount = 0
                self.status_var.set("Game started")
                update_controls_state()
                print("New game started!")

        def on_undo():
            try:
                self.board.pop()
                self.board.pop()
                self.display_board()
                self.log_var.set("Undo last two moves")
            except IndexError:
                self.log_var.set("No moves to undo")

        def on_red():
            self.game_start = False
            update_controls_state()
            self.status_var.set("Game stopped")
            print("Game stopped.")

        def toggle_orientation():
            if not self.game_start:
                orientation.set("w" if orientation.get() == "b" else "b")
                if self.sidePlayer == "w":
                    self.sidePlayer, self.sideRobot = "b", "w"
                else:
                    self.sidePlayer, self.sideRobot = "w", "b"
                self.log_var.set(f"Switched sides. Player: {'white' if self.sidePlayer == 'w' else 'black'}, Robot: {'white' if self.sideRobot == 'w' else 'black'}")

        def on_end():
            self.forfeitGame = True
            self.game_start = False
            self.game_end = True
            self.status_var.set("Game ended")
            print("Game ended by user.")
            update_controls_state()
            root.destroy()
        
        def show_help():
            help_window = tk.Toplevel(root)
            help_window.title("Help")
            help_window.geometry("450x300")
            
            title = "Chess Robot Instructions:"
            label = tk.Label(help_window, text=title, justify="center",
                             font=("Arial", 12, "bold"), padx=20, pady=10)
            label.pack(fill="both", expand=True, padx= 20)
            
            instructions = (
                "- Start New: Begin a new game (only works when stopped).\n"
                "- Undo: Undo the 2 previous moves of user and robot.\n"
                "- Stop: Stop the current game.\n"
                "- Switch Side: Flip player colour (only works when stopped).\n"
                "- End Game: Terminate the game loop and close the window.\n\n"
                "Notes:\n"
                "- The board updates live whenever moves are made.\n"
                "- Status and log messages are displayed below the board."
            )

            label = tk.Label(help_window, text=instructions, justify="left",
                             font=("Arial", 11), padx=20, pady=0)
            label.pack(fill="both", expand=True, padx= 20)

        def apply_engine_settings():
            try:
                elo_value = int(elo_entry.get())
                if 1350 <= elo_value <= 2850:
                    self.engine.configure({"UCI_LimitStrength": True,
                                        "UCI_Elo": elo_value})
                    self.log_var.set(f"Elo set to {elo_value}")
                else:
                    self.log_var.set("Elo must be between 1350 and 2850")
            except ValueError:
                self.log_var.set("Invalid Elo value")

            skill_value = skill_var.get()
            self.engine.configure({"Skill Level": skill_value})
            self.log_var.set(self.log_var.get() + f" | Skill Level set to {skill_value}")

        # Buttons
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        green_btn = tk.Button(button_frame, text="Start New", bg="green", fg="white",
                            width=10, command=on_green)
        green_btn.pack(side=tk.LEFT, padx=5)

        undo_btn = tk.Button(button_frame, text="Undo", bg="orange", fg="white",
                            width=10, command=on_undo)
        undo_btn.pack(side=tk.LEFT, padx=5)

        red_btn = tk.Button(button_frame, text="Stop", bg="red", fg="white",
                            width=10, command=on_red)
        red_btn.pack(side=tk.LEFT, padx=5)

        switch_btn = tk.Button(button_frame, text="Switch Side", bg="gray", fg="white",
                            width=10, command=toggle_orientation)
        switch_btn.pack(side=tk.LEFT, padx=5)

        end_btn = tk.Button(button_frame, text="End Game", bg="black", fg="white",
                            width=10, command=on_end)
        end_btn.pack(side=tk.LEFT, padx=5)
        
        help_btn = tk.Button(button_frame, text="Help", bg="blue", fg="white",
                             width=10, command=show_help)
        help_btn.pack(side=tk.LEFT, padx=5)
        
        # --- Elo and Skill Level controls ---
        control_frame = tk.Frame(root)
        control_frame.pack(pady=5)

        tk.Label(control_frame, text="Elo (1350–2850):").pack(side=tk.LEFT, padx=5)
        elo_entry = tk.Entry(control_frame, width=6)
        elo_entry.insert(0, "1600")
        elo_entry.pack(side=tk.LEFT, padx=5)

        tk.Label(control_frame, text="Skill Level (1–20):").pack(side=tk.LEFT, padx=5)
        skill_var = tk.IntVar(value=10)
        skill_slider = tk.Scale(control_frame, from_=1, to=20,
                                orient=tk.HORIZONTAL, variable=skill_var)
        skill_slider.pack(side=tk.LEFT, padx=5)

        apply_btn = tk.Button(control_frame, 
                              text="Apply", 
                              bg="#ff75c1", 
                              width=8, 
                              command=apply_engine_settings)
        apply_btn.pack(side=tk.LEFT, padx=10)
        
        # --- Helper to update widget states ---
        def update_controls_state():
            if self.game_start:
                elo_entry.config(state="disabled")
                skill_slider.config(state="disabled")
                apply_btn.config(state="disabled")
            else:
                elo_entry.config(state="normal")
                skill_slider.config(state="normal")
                apply_btn.config(state="normal")
        
        def draw_board():
            canvas.delete("all")
            for rank in range(8):
                for file in range(8):
                    x1 = file * SQUARE_SIZE
                    y1 = rank * SQUARE_SIZE
                    x2 = x1 + SQUARE_SIZE
                    y2 = y1 + SQUARE_SIZE
                    color = LIGHT if (file + rank) % 2 == 0 else DARK
                    canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=color)

            for square in chess.SQUARES:
                piece = self.board.piece_at(square)
                if piece:
                    file = chess.square_file(square)
                    rank = chess.square_rank(square)
                    if orientation.get() == "w":
                        draw_file = 7 - file
                        draw_rank = rank
                    else:
                        draw_file = file
                        draw_rank = 7 - rank
                    x = draw_file * SQUARE_SIZE + SQUARE_SIZE/2
                    y = draw_rank * SQUARE_SIZE + SQUARE_SIZE/2
                    
                    if piece.symbol() == "K" and self.board.is_check() and self.board.turn == chess.WHITE:
                        canvas.create_text(x, y, text=glyphs[piece.symbol()],
                                        font=("Arial", 32),
                                        fill="red")
                    elif piece.symbol() == "k" and self.board.is_check() and self.board.turn == chess.BLACK:
                        canvas.create_text(x, y, text=glyphs[piece.symbol()],
                                        font=("Arial", 32),
                                        fill="red")
                    else:
                        canvas.create_text(x, y, text=glyphs[piece.symbol()],
                                        font=("Arial", 32),
                                        fill="black")

        def refresh():
            draw_board()
            root.after(500, refresh)

        refresh()
        root.mainloop()


    # ---------------- Utility Methods ---------------- #
    
    # Create SSH client to SSH connect to Robot
    def create_SSH_client(self, server, port, user, password):
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(server, port, user, password)
            return client
        except TimeoutError:
            pass
    
    def send_SSH(self):
        with SCPClient(self.ssh.get_transport()) as scp:
            scp.put(self.commandSendDir, self.commandRobotDir)
            print('Move sent!')
    
    def get_SSH(self):
        with SCPClient(self.ssh.get_transport()) as scp:
            try:
                scp.get(self.statusRobotDir, self.statusGetDir)
                #print('Get status succesfully!')
            except SCPException:
                pass

    # -------------- Machine Vision ------------- #
    def capture_and_save(self, keyword, save_dir="captured_images", zoom_factor=1.2):
        """
        Capture a single image from the already-open webcam, apply optional zoom, and save it.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        for _ in range(5):
            self.cap.grab()
        ret, frame = self.cap.retrieve()    
        #ret, frame = self.cap.read()

        if not ret:
            self.cap.release()
            self.cap = cv2.VideoCapture(0)
            ret, frame = self.cap.read()

        # Apply zoom by cropping the center
        if zoom_factor > 1:
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            radius_x, radius_y = int(center_x / zoom_factor), int(center_y / zoom_factor)
            min_x, max_x = center_x - radius_x, center_x + radius_x
            min_y, max_y = center_y - radius_y, center_y + radius_y
            frame = frame[min_y:max_y, min_x:max_x]
            frame = cv2.resize(frame, (w, h))

        # Save with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"{keyword}_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Image saved: {filename}")
        return filename
    
    # ---------------- Game Flow ---------------- #
    def display_board(self, fen: str = None):
        """
        Display the current board. If a FEN string is provided, display that board
        side-by-side to the right of the current board.
        Supports vertical and horizontal flipping.
        """
        import chess

        def board_ascii(board: chess.Board, 
                        title: str, 
                        flip_v: bool = False, 
                        flip_h: bool = False):
            board_str = str(board)
            lines = board_str.split('\n')

            # Flip vertically (reverse ranks)
            if flip_v:
                lines = list(reversed(lines))

            # Flip horizontally (reverse files in each rank)
            if flip_h:
                lines = [' '.join(reversed(line.split())) for line in lines]

            top = [f'   {title}', '┌' + '─' * 17 + '┐']
            body = []
            for line in lines:
                squares = line.split()
                row = '│ ' + ' '.join(squares) + ' │'
                body.append(row)
            bottom = ['└' + '─' * 17 + '┘']
            return top + body + bottom

        # Decide flips based on robot side
        flip_v = (self.sideRobot == "w")   # vertical flip if robot is white
        flip_h = (self.sideRobot == "w")

        # Left: current board
        left_board = self.board
        left_lines = board_ascii(left_board, "CURRENT BOARD", flip_v, flip_h)

        if not fen:
            for line in left_lines:
                print(line)
            return

        # Right: FEN board
        right_board = chess.Board(fen)
        right_lines = board_ascii(right_board, "FEN BOARD", flip_v, flip_h)

        spacer = "   "
        for l, r in zip(left_lines, right_lines):
            print(f"{l:<30}{spacer}{r}")
        
    def get_best_move(self, time=0.5, depth=5):
        # Force en passant
        """for move in self.board.legal_moves:
            if self.board.is_en_passant(move):
                return str(move)"""
        
        result = self.engine.play(self.board, chess.engine.Limit(time=time, depth=depth))
        print(f"Engine move: {result.move}")
        return str(result.move)

    def make_command_file(self, side: str, move: str):
        # Calculate coordinates for captured pieces
        if move != None:
            if self.board.is_capture(chess.Move.from_uci(move)):
                if self.capturedBatch != 0:
                    # Calculate coordinate for captured piece
                    # Vertical (y) direction > Horizontal (x) direction
                    self.capturedY = (self.capturedAmount % self.capturedBatch) * self.coefY + self.initialY
                    self.capturedX = int(self.capturedAmount / self.capturedBatch) * self.coefX + self.initialX
                self.capturedAmount += 1    # Increase the amount of captured pieces by 1
            print(f'\tCaptured amount: {self.capturedAmount}')
            print(f'\tCaptured coordinates: {self.capturedX, self.capturedY}')
            
            with open(self.commandSendDir, 'w') as file:
                sMove = '0000'
                # Castling
                if self.board.is_castling(chess.Move.from_uci(move)):
                    row = move[1]   # Extract row of castling
                    if self.board.is_kingside_castling(chess.Move.from_uci(move)):
                        sMove = 'h' + row + 'f' + row
                    else:
                        sMove = 'a' + row + 'd' + row
                    typeMove = 's'
                # Capture
                elif self.board.is_capture(chess.Move.from_uci(move)):
                    if self.board.is_en_passant(chess.Move.from_uci(move)):
                        typeMove = 'e'
                    else:
                        typeMove = 'c'
                else:
                    typeMove = 'm'
                command_lines = [side, typeMove, move, sMove,
                                str(self.capturedX), str(self.capturedY)]
                for idx, line in enumerate(command_lines):
                    command_lines[idx] = line + '\n'
                
                file.writelines(command_lines)
            print('Robot in action..')
    
    def choose_side(self):
        # Player picks side
        while self.sidePlayer not in ('w', 'b', 'c'):
            self.sidePlayer = input('Type "w" to play on white, "b" to play on black, or "c" to cancel: ')

        if self.sidePlayer == 'c':
            print('Game cancelled')
            return False
        else:
            self.sideRobot = 'b' if self.sidePlayer == 'w' else 'w'
            return True
    
    def player_make_move(self):
        # Play enters a move when Robot is "idle"
        statusRobot = 'busy'
        while statusRobot == 'busy':
            self.log_var.set("Robot in action..")
            self.get_SSH()
            with open(self.statusGetDir) as file:
                statusRobot = file.read().replace('\n', '')
        self.log_var.set("Robot in action..")
        print(f'Get status successfully: {statusRobot}')
        movePlayer = ''
        validMove = False
        while not validMove: 
            
            if not self.game_start:
                break
            
            # Check if Robot is 'idle'
            if statusRobot != 'busy':
                filename = self.capture_and_save(keyword='live')
                print(filename)
                if not self.read_corner_once:
                    pieces, self.corners = self.detector.predict(filename, save= False, print_result= False)
                    if len(self.corners) == 4:
                        self.read_corner_once = True
                
                pieces, corners = self.detector.predict(filename, save= False, print_result= False)
                
                if len(corners) == 4:
                    self.corners = corners
                
                pieces = self.detector.resolve_queen_king_conflict(pieces, 0.5, 0.25)
                # pieces = self.detector.resolve_black_queen_king_conflict(pieces, 0.5, 0.25)
                boardMatrix, squareMap = self.detector.build_board_with_pieces(pieces, 
                                                                               self.corners if len(corners) != 4 else corners, 
                                                                               self.sideRobot)
                newBoard = self.detector.board_to_fen(boardMatrix, self.sideRobot)
                
                if not self.game_end:
                    if self.fen_difference_to_uci(newBoard) is None and chess.Board(newBoard).board_fen() != self.board.board_fen():
                        self.log_var.set("Illegal move or incorrect board recognition")
                    else:
                        self.log_var.set("")
                    
                # Delete new photo if the same board is detected, otherwise update previous board to the new one
                if newBoard != self.prevBoard and self.fen_difference_to_uci(newBoard):
                    self.prevBoard = newBoard
                else:
                    if os.path.exists(filename):
                        os.remove(filename)
                        # pass
                        
                #print(f'New Board: {newBoard}')
                self.display_board(newBoard)
                
                print(f'Conversion: {self.fen_difference_to_uci(newBoard)}')
                
                movePlayer = self.fen_difference_to_uci(newBoard)       # Get move from Player
                    
            if movePlayer != None:
                # Check for legal moves
                try:
                    if chess.Move.from_uci(movePlayer) in self.board.legal_moves:
                        validMove = True
                except ValueError:
                    pass
            else:
                #time.sleep(0.1)
                pass
        return movePlayer
                
    def fen_difference_to_uci(self, fen_after: str):
        """
        Compare self.board (current state) with fen_after.
        Return the UCI move that transforms self.board into fen_after,
        considering only the piece placement (ignore side, castling, etc.).
        If no single move matches, return None.
        """
        # Copy current board state
        board_before = chess.Board(self.board.fen())
        board_after = chess.Board(fen_after)

        target_position = board_after.board_fen()  # only piece placement

        for move in board_before.legal_moves:
            board_before.push(move)
            if board_before.board_fen() == target_position:
                return move.uci()
            board_before.pop()

        return None
    
    def compare_fen(self, fen_after: str):
        """
        Compare self.board (current state) with fen_after.
        Return True if 2 FENs are identical, otherwise False
        """
        # Copy current board state
        board_before = chess.Board(self.board.fen())
        board_after = chess.Board(fen_after)

        if board_after.board_fen() ==  board_before.board_fen():
            return True
        else:
            return False


    ## MAIN PROGRAM ##
    def play(self):
        os.system('cls' if os.name == 'nt' else 'clear')

        # Main loop
        while not self.game_end:
            if not self.game_start:
                continue

            if self.board.is_game_over(claim_draw=True):
                self.game_start = False
                self.status_var.set('Game over')
                print("Game over!")
            
            if self.sidePlayer == 'b' and not self.firstMove:  # Player on black, engine on white plays first
                while 1:
                    if self.game_end:
                        self.game_start = False
                        break
                    
                    filename = self.capture_and_save(keyword='live')
                    pieces, corners = self.detector.predict(filename, save= False, print_result= False)
                    if len(corners) == 4:
                        self.corners = corners
                        self.read_corner_once = True
                    boardMatrix, squareMap = self.detector.build_board_with_pieces(pieces, 
                                                                                   self.corners if len(corners) != 4 else corners, 
                                                                                   self.sideRobot)
                    newBoard = self.detector.board_to_fen(boardMatrix, self.sideRobot)
                    self.display_board(newBoard)
                    if not self.compare_fen(newBoard):
                        time.sleep(0.1)
                        if not self.firstPrint:
                            self.log_var.set("Wrong board placement!")
                            print('Wrong board placement!')     
                    else:
                        break
                               
                    
                move = self.get_best_move()                
                self.make_command_file(self.sideRobot, move)
                self.board.push(chess.Move.from_uci(move))
                self.send_SSH()
                self.display_board()
                self.firstMove = True                          # Set first move variable to True so Robot only plays first once
                
            # Player makes the move
            move = self.player_make_move()
            if self.game_start:
                self.board.push(chess.Move.from_uci(move))      # Acknowledged Player's legal move
            
                if self.board.is_game_over():
                    self.log_var.set('Player won!')
                    self.status_var.set('Game over')
                    print('Player won!')
                    self.game_start = False
            
            if self.game_start:
                # Engine makes the move
                move = self.get_best_move()                
                self.make_command_file(self.sideRobot, move)
                self.send_SSH()
                self.board.push(chess.Move.from_uci(move))
                self.display_board()
            
        print("Game exit")
        self.close()

    def close(self):
        self.engine.quit()
        self.ssh.close()
        if self.cap:
            self.cap.release()
            print("Webcam released.")


# ---------------- Run if Main Program ---------------- #
if __name__ == "__main__":
    engine_path = "stockfish/stockfish-windows-x86-64-avx2.exe"
    game = ChessGame(engine_path)
    # Launch GUI in a separate thread
    import threading
    threading.Thread(target=game.play, daemon=True).start()
    game.launch_board_gui()
