def pytest_addoption(parser):
    """Adds the --game-name command-line option to pytest."""
    parser.addoption(
        "--game-name", 
        action="store", 
        required=True, 
        help="Name of the game to test (e.g., 'pong')."
    )