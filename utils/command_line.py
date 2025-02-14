"""A module to provide an interface to the subprocess library.

This module hosts the functionality for streamline the execution of command-line
algorithms called from the subprocess library. This is convenient when parsing through
gear configuration parameters and inputs specified in a gear configuration 
(e.g `config.json`).

Examples:
    >>> command=["ls"]
    >>> ParamList={"l":True,"a":True,"h":True}
    >>> command = build_command_list(command, ParamList)
    >>> command
    ["ls", "-l", "-a", "-h"]
    >>> exec_command(command)

will both build a command-line list and execute it with the subprocess
Popen command. Providing `stdout`, `stderr`, and raising an `exception` on
non-zero exit from the command.

    >>> command = ["du"]
    >>> params = {"a": True, "human-readable": True, "max-depth": 3}
    >>> command = build_command_list(command, params)
    >>> command
    ["du", "-a", "--human-readable", "--max-depth=3"]
    >>> params = {"dir1":".","dir2":"/tmp"}
    >>> command = build_command_list(command, params, include_keys=False)
    >>> command
    ["du", ".", "/tmp"]
    >>> exec_command(command)

The above are examples that demonstrate how command-line parameters can be
built from gear configuration options.
"""

import logging
import subprocess as sp

log = logging.getLogger(__name__)


def _remove_prohibited_values(param_list):
    """
    Removes `None` and empty strings from the param_list.

    Args:
        param_list (dict): Dictionary of parameter values.

    Returns:
        dict: Dictionary of parameter values with `None` values and empty strings
            removed.
    """
    param_list_new = param_list.copy()
    for key, value in param_list.items():
        if not value or value == "":
            param_list_new.pop(key)
            log.warning('Removing parameter with empty value for key "%s".', key)
    return param_list_new


def build_command_list(command, param_list, include_keys=True):
    """Build a command line list to be used by subprocess.Popen command.

    Args:
        command (list): A list containing the base command (e.g. ["ls"]) with
            parameters that are always used.
        param_list (dict): An dictionary (usually ordered) of key/value pairs
            representing command-line parameters/switches for the command in
            question. Results in ("-k value" or "--key=value"). If the value is boolean,
            presence/absence of the key (with out value) is determined by a True/False
            value. Values of `None` or empty strings will be interpretted as False.
        include_keys (bool, optional): A flag to indicate whether or not to include the
            keys in the command list. Defaults to True.

    Returns:
        list: returns the completed command-list expected by subprocess.Popen.

    Example:
        >>> command = ["du"]
        >>> params = {"a": True, "human-readable": True, "max-depth": 3}
        >>> command = build_command_list(command, params)
        >>> command
        ["du", "-a", "--human-readable", "--max-depth=3"]
    """
    param_list = _remove_prohibited_values(param_list)
    for key in param_list.keys():
        # Single character command-line parameters are preceded by a single "-"
        if len(key) == 1:
            # If Param is boolean and true, include, else exclude
            if isinstance(param_list[key], bool) and param_list[key]:
                command.append("-" + key)
            else:
                if include_keys:
                    command.append("-" + key)
                if str(param_list[key]):
                    command.append(str(param_list[key]))
        # Multi-Character command-line parameters are preceded by a double "--"
        else:
            # If Param is boolean and true include, else exclude
            if isinstance(param_list[key], bool):
                if param_list[key] and include_keys:
                    command.append("--" + key)
            else:
                item = ""
                if include_keys:
                    item = "--" + key
                    item = item + "="
                item = item + str(param_list[key])
                command.append(item)
    return command


def exec_command(
    command,
    dry_run=False,
    environ=None,
    shell=False,
    stdout_msg=None,
    cont_output=False,
):
    """
    An abstraction to execute prepared shell commands using the subprocess module.

    Args:
        command (list): list of command-line parameters, starting with the
            command to run.
        dry_run (bool, optional): a boolean flag to indicate a dry-run without
            executing anythingj. Defaults to False.
        environ (dict, optional): a dictionary of key/value pairs representing
            the environment variables necessary for running the command-line
            utility. Defaults to an empty dictionary {}.
        shell (bool, optional): whether or not to execute as a single
            shell string. This facilitates output redirects. Defaults to False.
        stdout_msg (string, optional): A string to notify the user where the
            stdout/stderr has been redirected to. Defaults to None.
        cont_output (bool, optional): Used to provide continuous output of
            stdout without waiting until the completion of the shell command.
            Defaults to False.
    Returns:
        stdout, stderr, returncode
    Raises:
        RuntimeError: If the return value from the command-line function is not zero.

    Example:
            >>> command = ["du"]
            >>> params = {"a": True, "human-readable": True, "max-depth":3}
            >>> command = build_command_list(command, params)
            >>> params = {"dir1":".","dir2":"/tmp"}
            >>> command = build_command_list(command, params, include_keys=False)
            >>> exec_command(command)
    """
    log.info("Executing command: \n %s \n\n", " ".join(command))
    if not dry_run:
        # The "shell" parameter is needed for bash output redirects
        # (e.g. >,>>,&>)

        result = sp.Popen(
            command,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            universal_newlines=True,
            env=environ,
            shell=shell,
        )

        # log that we are using an alternate stdout message
        if stdout_msg is not None:
            log.info(stdout_msg)

        # if continuous stdout is desired... and we are not redirecting output
        if cont_output and not (shell and (">" in command)) and (stdout_msg is None):
            while True:
                stdout = result.stdout.readline()
                if stdout == "" and result.poll() is not None:
                    break
                if stdout:
                    print(stdout.rstrip())

            returncode = result.poll()
            stderr = "".join(result.stderr.readlines())
        else:
            stdout, stderr = result.communicate()

            returncode = result.returncode
            if stdout_msg is None:
                log.info(stdout)

        log.info("Command return code: %s", returncode)

        if returncode != 0:
            log.error(stderr)
            raise RuntimeError("The following command has failed: \n{}".format(command))

        return stdout, stderr, returncode
