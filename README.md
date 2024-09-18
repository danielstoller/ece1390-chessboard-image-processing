# Chessboard Recognition Project

## Description

This project aims to develop a program capable of locating and extracting a chessboard from a still image. Using advanced image processing techniques, the program will identify the pieces on the board, determine their positions, and convert the game state into a chess code (e.g., FEN). This allows players to easily export the game into a chess engine for analysis. The program is beneficial for over-the-board chess players who want a quick way to analyze their games by taking a photo and uploading it for review. Additionally, it can help users resume incomplete games or serve as an interface for robotic chess arms that use chess engines to make moves.

## Code Specifications

- **Input**: JPG images of various chess positions. For future expansion, video capture at 60 FPS can be used. The only other required input is the player’s turn.
- **Output**: Tags identifying each piece on the board and the corresponding FEN code, which can be exported into a chess engine for further analysis.
- **Additional Requirements**: The project can be extended to real-time video processing or image capture automation as a stretch goal.

## Planned Approach

1. **Image Capture and Automation (Stretch Goal)**: Automating image capture and upload via a button-triggered camera module that uploads the image for processing.
2. **Image Filtering**: Convert the image to grayscale, apply Gaussian blur to remove noise, and adjust brightness and contrast.
3. **Corner Detection and Grid Segmentation**: Using Harris Corner Detection or Hough Transform to detect chessboard corners and segment the board into an 8x8 grid.
4. **Piece Identification**: Use template matching and contour detection to distinguish different chess pieces and a pre-trained model to classify piece types.
5. **Data Conversion**: Translate identified pieces into FEN or PGN format, which can be understood by chess engines.
6. **Chess Engine Integration**: Pass the board’s state to a chess engine (like Stockfish) for move evaluation and opponent play.
7. **Automated Move Output (Stretch Goal)**: Integrate with a physical chess interface or robotic arm to execute the next move.

## Timeline

- **Sep 18**: Finalize White Paper and project specifications
- **Sep 20**: Split and assign work between group members
- **Sep 23**: Decide on filtering strategy
- **Oct 2**: Complete filtering code and testing
- **Oct 4**: Decide on edge detection strategy
- **Oct 9**: Decide on piece identification strategy
- **Oct 15**: Finalize edge detection code
- **Nov 4**: Finalize piece identification
- **Nov 18**: Finalize data conversion
- **Dec 2**: Complete chess engine integration
- **Dec 2-9**: Final testing and work on stretch goals
- **Dec 9, 2024**: Project due

## Metrics of Success

1. **Board-edge detection**: Successful identification of chessboard edges and 8x8 segmentation, even with pieces on the board.
2. **Chess Piece Identification**: Accurate distinction of piece types and colors, as well as their correct positioning on the board.
3. **Data Conversion**: FEN or PGN representation of the game state must accurately reflect the board, with correct piece locations and types.

## Pitfalls and Alternative Solutions

- **Piece Identification**: We may encounter difficulties in identifying pieces. If existing code doesn’t work, we’ll need to modify or rewrite it.
- **Piece Positioning**: Pieces not centered on squares may cause misclassification. We’ll need to develop error-correction mechanisms to handle this.
- **Image Capture Automation**: Automating image uploads might prove difficult, and if we cannot achieve this, manual image uploads will serve as a fallback.
