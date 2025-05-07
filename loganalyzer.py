from mrjob.job import MRJob
import re

LOG_PATTERN = r'\[(.?)\] \[(.?)\] \[(.?)\] (.)'

class LogAnalyzer(MRJob):
    def mapper(self, _, line):
        match = re.match(LOG_PATTERN, line)
        if match:
            timestamp, level, user, message = match.groups()
            yield (level.strip(), 1)

    def reducer(self, level, counts):
        yield (level, sum(counts))

if __name__ == '_main_':
    LogAnalyzer.run()
    