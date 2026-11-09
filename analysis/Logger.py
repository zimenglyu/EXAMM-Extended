class Logger:
    INFO = 1
    NONE = 0
    DEBUG = 2
    ERROR = 3

    LEVELS = {
        'INFO': INFO,
        'NONE': NONE,
        'DEBUG': DEBUG,
        'ERROR': ERROR,
    }

    def __init__(self, level='INFO'):
        """
        Initialize the logger with a logging level.

        :param level: A string representing the logging level. Defaults to 'INFO'.
        """
        self.level = self.LEVELS.get(level, self.INFO)

    def log(self, message, level='INFO'):
        """
        Log a message if the current level is appropriate.

        :param message: The message to log.
        :param level: The level of the message being logged.
        """
        if self.level >= self.LEVELS[level]:
            print(message)

    def set_level(self, level):
        """
        Set the logging level.

        :param level: A string representing the new logging level.
        """
        self.level = self.LEVELS.get(level, self.INFO)

# Example usage:
# logger = Logger()  # Defaults to INFO level

# logger.log("This is an info message.", level='INFO')
# logger.log("This is an error message.", level='ERROR')

# logger.set_level('NONE')  # Set logging level to NONE
# logger.log("This message will not be printed.", level='INFO')

# logger.set_level('DEBUG')  # Set logging level to DEBUG
# logger.log("Debugging session starts.", level='DEBUG')
