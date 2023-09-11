import sys
import platform
import pickle
from bz2 import BZ2File

# http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113
class ProgressBar():
    DEFAULT_BAR_LENGTH = 50
    DEFAULT_CHAR_ON  = '>'
    DEFAULT_CHAR_OFF = ' '

    def __init__(self, end, start=0, name='N/A'):
        self.end    = end
        self.start  = start
        self.name = name
        self._barLength = self.__class__.DEFAULT_BAR_LENGTH

        self.setLevel(self.start)
        self._plotted = False

    def setLevel(self, level):
        self._level = level
        if level < self.start:  self._level = self.start
        if level > self.end:    self._level = self.end

        self._ratio = float(self._level - self.start) / float(self.end - self.start)
        self._levelChars = int(self._ratio * self._barLength)

    def plotProgress(self):
        tab = '\t'
        sys.stdout.write("\r%s%3i%% [%s%s] (%s)" %(
            tab*1 + '  ', int(self._ratio * 100.0),
            self.__class__.DEFAULT_CHAR_ON  * int(self._levelChars),
            self.__class__.DEFAULT_CHAR_OFF * int(self._barLength - self._levelChars),
            self.name
        ))
        sys.stdout.flush()
        self._plotted = True

    def setAndPlot(self, level):
        oldChars = self._levelChars
        self.setLevel(level)
        if (not self._plotted) or (oldChars != self._levelChars):
            self.plotProgress()

    def __add__(self, other):
        assert type(other) in [float, int], "can only add a number"
        self.setAndPlot(self._level + other)
        return self

    def __sub__(self, other):
        return self.__add__(-other)

    def __iadd__(self, other):
        return self.__add__(other)

    def __isub__(self, other):
        return self.__add__(-other)

    def finish(self):
        sys.stdout.write("\n")


def load_from_dmp(dmp_path):
    """
    Dump all function information collected from IDA Pro
    Each function represents an instance of class unit.IDA_Function()
    :param dmp_path:
    :return:
    """
    functions = dict()
    dmp_file = BZ2File(dmp_path, 'rb')
    cnt = 0
    while True:
        try:

            major_ver, _, _ = platform.python_version_tuple()
            if major_ver == '2':
                import cPickle
                #F = cPickle.load(dmp_file)
                F = pickle.load(dmp_file)
            if major_ver == '3':
                F = pickle.load(dmp_file, encoding='latin1')

            if not F:
                break
            functions[F.start] = F
            cnt += 1

        except MemoryError:
            logging.error('Memory error reading at Function 0x%08X after loading %d functions'
                          % (F.addr, cnt))
            pass

    dmp_file.close()
    return functions
