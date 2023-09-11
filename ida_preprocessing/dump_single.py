import os, sys
import logging
import subprocess
 
def call_ida(input_file, ida_path=None):
    """
    :param input_file:
    :param ida_path:
    :return:
    """
    def _check_ida():
        for pgm in [x for x in os.environ['PATH'].split(';')]:
            if 'ida' in pgm.lower():
                return True
        return False

    # Check if IDA Pro has been installed in Windows
    if _check_ida() or ida_path:
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dmp.py")
        if not os.path.exists(script):
            logging.error("Error: could not find ida.py (%s)" % script)
            sys.exit(1)
        if ida_path:
            ida_path += 'ida64.exe'
        command = ida_path + ' -A -S"\\"' + script + '\\"" ' + input_file
        print(command)
        logging.info("[+] Executing: %s" % command)
        res = subprocess.call(command)
        #res = subprocess.Popen(command, shell=True)
        #res.kill()
        if res == 0:
            logging.info("[+] Dumped analyses successfully with IDA Pro...")
            print ("[+] Dumped analyses successfully with IDA Pro...")
        else:
            logging.error("[-] Something went wrong while invoking IDA Pro")
            print("[-] Something went wrong while invoking IDA Pro")
    else:
        logging.info("[-] Fail to find IDA Pro")


if __name__ == "__main__":
    # This is the path of your input binary to dump
    input_path = "sample_binary\gcc-5__Ou__direvent__direvent"
    # This will be the path of your IDA
    ida_path = "C:\\Users\\...\\IDAPro7.5\\"
    call_ida(input_path, ida_path)