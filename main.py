import os
import sys

# We do not use stcli as it can cause problems when not run through streamlit directly.
# This script serves as a simple pointer.
if __name__ == "__main__":
    print("This is the main entrypoint.")
    print("Please run the application using:")
    print("    streamlit run app/main.py")
    sys.exit(0)
