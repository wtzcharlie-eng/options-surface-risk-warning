"""Read-only current document check; old numeric-string checker is archived."""
import sys
from build_submission import main
if __name__=='__main__':
    sys.argv=[sys.argv[0],'--check'];main()
