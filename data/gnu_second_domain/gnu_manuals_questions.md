# GNU Manuals Evaluation — Question Set (Review Copy)

Total questions: **135** across three sub-corpora. Every `expected_pages` value is a **PDF physical page** (1-based, as opened in a viewer or produced by `pdftoppm -f N`), verified by locating the answer text in that page of the specific uploaded PDF.

## GNU Bash Reference Manual  (`corpus=bash`  ≙ WMP PG&E)
Source: `gnu-bash-reference-manual.pdf` · 214 pages · 30 in-scope + 7 negative = 37

| qid | pages | type | question | reference answer |
|---|---|---|---|---|
| bash-001 | 82 | concept | What does the pipefail option do and how is it enabled? | With pipefail enabled (set -o pipefail), a pipeline's return status is the value of the last (rightmost) command to exit non-zero, or zero if all succeed. Without it, a pipeline's status is just that of the last command. |
| bash-002 | 54 | lookup | What is the value range of a normal command exit status in bash? | Exit statuses fall between 0 and 255. The shell uses values above 125 specially (e.g., 126 not executable, 127 not found, 128+N killed by signal N). |
| bash-003 | 30 | lookup | Which special parameter holds the exit status of the most recently executed foreground pipeline? | $? expands to the exit status of the most recently executed foreground command/pipeline. |
| bash-004 | 103 | lookup | What does the $RANDOM variable expand to? | Each reference to $RANDOM expands to a random integer between 0 and 32767. Assigning a value seeds the generator. |
| bash-005 | 116 | lookup | How do you declare an associative array in bash? | Use declare -A name to create an associative array (declare -a creates an indexed array). Associative arrays are indexed by arbitrary string keys. |
| bash-006 | 12 | concept | Inside double quotes, does the dollar sign keep its special meaning? | Yes. Within double quotes, $, backtick, and backslash retain special meaning (so parameter/command/arithmetic expansion still happen); inside single quotes nothing is expanded. |
| bash-007 | 84 | concept | What does the shopt builtin do? | shopt toggles optional shell behavior settings: -s sets, -u unsets, -p prints them; with -o it operates on set -o options. |
| bash-008 | 65 | concept | What does the trap builtin do? | trap [-lpP] [action] [sigspec ...] arranges for action to be read and executed when the shell receives the listed signals; special names like EXIT, ERR, and DEBUG are also supported. |
| bash-009 | 117 | lookup | What does ${#parameter} expand to? | It substitutes the length in characters of the value of parameter; for arrays with @ or * it gives the number of elements. |
| bash-010 | 35 | lookup | How do you extract a substring with parameter expansion? | Use ${parameter:offset:length} (Substring Expansion): it expands to up to length characters of parameter starting at offset (offset may be negative with a leading space). |
| bash-011 | 34 | concept | What does ${parameter:-word} do? | If parameter is unset or null, the expansion of word is substituted; otherwise the value of parameter is used. Omitting the colon tests only for unset (not null). |
| bash-012 | 42 | lookup | What is the syntax for command substitution? | $(command) (or the older backtick form) runs command and replaces the construct with its standard output, with trailing newlines removed. |
| bash-013 | 43 | concept | What is process substitution? | Process substitution, written <(list) or >(list), lets a process's input or output be referred to as a filename; bash connects it via a pipe or named FIFO. |
| bash-014 | 27 | concept | What does the local builtin do inside a function? | local declares variables that are visible only within the function and its children; without it, variables are shared with the caller. |
| bash-015 | 76 | lookup | Which read options set a prompt and a timeout? | read -p prompt prints prompt before reading (no trailing newline); read -t timeout limits how long read waits for input. |
| bash-016 | 50 | concept | What is a here-string? | A here-string, [n]<<< word, feeds the expanded word (with a trailing newline) to the command's standard input; it is a compact variant of a here-document. |
| bash-017 | 49 | lookup | How do you redirect standard error to standard output? | Use 2>&1 (place it after any stdout redirection, e.g., > file 2>&1). The \|& operator is shorthand for 2>&1 \| when piping. |
| bash-018 | 31 | concept | What does brace expansion produce? | Brace expansion generates arbitrary strings: a{b,c}d yields abd acd, and {1..5} yields a numeric/character sequence. It is purely textual and happens before other expansions. |
| bash-019 | 33 | concept | What does tilde expansion do? | A leading unquoted ~ expands to $HOME, ~user to that user's home directory, ~+ to $PWD, and ~- to $OLDPWD. |
| bash-020 | 61 | concept | What does the getopts builtin do? | getopts optstring name parses positional parameters for single-character options; optstring lists valid option letters (a trailing colon means the option takes an argument, stored in OPTARG). |
| bash-021 | 92 | concept | What does the IFS variable control? | IFS (Internal Field Separator) is the set of characters used to split words during expansion and by the read builtin when splitting a line into fields. |
| bash-022 | 80 | concept | What does set -e (errexit) do? | With errexit, the shell exits immediately if a command (subject to the documented exceptions) returns a non-zero status. |
| bash-023 | 18 | lookup | What is the syntax of a C-style for loop in bash? | for (( expr1 ; expr2 ; expr3 )) ; do commands ; done — it evaluates the arithmetic expressions like C, running commands while expr2 is non-zero. |
| bash-024 | 20 | concept | What does the select compound command do? | select name [in words ...]; do commands; done generates a numbered menu from words, reads a choice into name (using PS3), and loops until end-of-file. |
| bash-025 | 75 | concept | What does the printf builtin do? | printf [-v var] format [arguments] writes the arguments formatted under control of a printf-style format string; -v assigns the result to a variable instead of printing. |
| bash-026 | 134 | concept | What does the wait builtin do? | wait [-fn] [-p varname] [id ...] waits for the specified jobs/processes to finish and returns the exit status of the last one; with no id it waits for all background children. |
| bash-027 | 92 | lookup | What is the CDPATH variable used for? | CDPATH is a colon-separated list of directories that cd searches when its argument is not an absolute path. |
| bash-028 | 66 | concept | What does the umask builtin set? | umask [-p] [-S] [mode] sets the file-creation mask (octal or symbolic); -S prints it symbolically and -p in a reusable form. |
| bash-029 | 67 | concept | What does the alias builtin do? | alias [name[=value] ...] defines or lists aliases (word substitutions applied as the first word of a simple command); with -p or no args it prints current aliases. |
| bash-030 | 60 | concept | What does the exec builtin do? | exec [-cl] [-a name] [command ...] replaces the shell with command without creating a new process; with no command it applies redirections to the current shell. |
| bash-neg-01 | — | negative_cross_doc | What does the .PHONY target do? | Not answerable from the bash manual. Phony targets are a GNU make concept, documented in the make manual. A correct response should decline and state the topic is not covered by this document. |
| bash-neg-02 | — | negative_cross_doc | What does the automatic variable $< refer to? | Not answerable from the bash manual. Automatic variables like $< belong to GNU make, not bash. A correct response should decline and state the topic is not covered by this document. |
| bash-neg-03 | — | negative_cross_doc | How do you verify a file's SHA-256 checksum? | Not answerable from the bash manual. sha256sum is a coreutils program, not a bash feature. A correct response should decline and state the topic is not covered by this document. |
| bash-neg-04 | — | negative_cross_doc | What does the patsubst function do? | Not answerable from the bash manual. patsubst is a GNU make text function. A correct response should decline and state the topic is not covered by this document. |
| bash-neg-05 | — | negative_out_of_scope | How do you configure nginx as a reverse proxy? | Not answerable from the bash manual. This concerns the nginx web server and is outside all three GNU manuals. A correct response should decline and state the topic is not covered by this document. |
| bash-neg-06 | — | negative_out_of_scope | What is the time complexity of quicksort? | Not answerable from the bash manual. This is an algorithms question unrelated to the manual. A correct response should decline and state the topic is not covered by this document. |
| bash-neg-07 | — | negative_out_of_scope | How do you train a convolutional neural network in PyTorch? | Not answerable from the bash manual. This is a machine-learning topic outside the manual. A correct response should decline and state the topic is not covered by this document. |

## GNU Coreutils Manual  (`corpus=coreutils`  ≙ WMP SCE)
Source: `gnu-coreutils-manual.pdf` · 319 pages · 40 in-scope + 8 negative = 48

| qid | pages | type | question | reference answer |
|---|---|---|---|---|
| cu-001 | 12 | lookup | What does the --help option print for GNU coreutils programs? | --help prints a usage message listing all available options and then exits successfully. |
| cu-002 | 24 | lookup | What does cat do with the -n option? | cat -n (--number) numbers all output lines, starting at 1. |
| cu-003 | 24 | lookup | What does cat do with the -s option? | cat -s (--squeeze-blank) suppresses repeated adjacent blank lines, leaving a single blank line. |
| cu-004 | 25 | concept | What does the tac command do? | tac concatenates files and writes them to standard output with lines in reverse order (last line first). |
| cu-005 | 41 | lookup | How many lines does head print by default and how do you change it? | head prints the first 10 lines of each file by default; -n N (--lines) prints N lines instead. |
| cu-006 | 43 | lookup | How do you make tail follow a growing file? | tail -f (--follow) keeps the file open and outputs data as it is appended, useful for watching logs. |
| cu-007 | 53 | lookup | What three counts does wc print by default? | By default wc prints the newline, word, and byte counts for each file. |
| cu-008 | 53 | lookup | How do you print only the byte count with wc? | wc -c (--bytes) prints the number of bytes. |
| cu-009 | 63 | lookup | How do you sort numerically with sort? | sort -n (--numeric-sort) compares lines by leading numeric value rather than lexically. |
| cu-010 | 65 | lookup | How do you reverse the sort order? | sort -r (--reverse) reverses the result of the comparisons. |
| cu-011 | 62 | lookup | How do you sort on a specific key field? | sort -k KEYDEF sorts using the specified field range (e.g., -k2,2 sorts on the second field); -t sets the field separator. |
| cu-012 | 62 | lookup | How do you output only unique lines with sort? | sort -u (--unique) outputs only the first line of each group of equal lines. |
| cu-013 | 73 | lookup | What does uniq -c do? | uniq -c (--count) prefixes each output line with the number of times it occurred among adjacent duplicates. |
| cu-014 | 87 | lookup | How do you select specific fields with cut? | cut -f LIST (--fields) selects the listed fields, split by the delimiter set with -d (default TAB). |
| cu-015 | 86 | lookup | How do you select specific characters with cut? | cut -c LIST (--characters) selects only the characters at the listed positions. |
| cu-016 | 88 | concept | What does the paste command do? | paste merges files line by line, writing corresponding lines from each file joined by TAB (or the -d delimiter). |
| cu-017 | 90 | concept | What does the join command do? | join writes a line for each pair of input lines from two sorted files that share a common join field. |
| cu-018 | 59 | lookup | How do you verify checksums with sha256sum? | sha256sum -c (--check) reads a list of checksums and filenames and verifies each file against its recorded digest. |
| cu-019 | 46 | lookup | How do you split a file into pieces of N lines? | split -l N (--lines) writes N lines per output file. |
| cu-020 | 46 | lookup | How do you split a file by byte size? | split -b SIZE (--bytes) writes SIZE bytes per output file (SIZE may use suffixes like K, M). |
| cu-021 | 75 | concept | What does the comm command do? | comm compares two sorted files line by line and outputs three columns: lines only in file1, only in file2, and common to both. |
| cu-022 | 97 | lookup | How do you delete characters with tr? | tr -d SET1 deletes all characters that appear in SET1 from the input. |
| cu-023 | 97 | lookup | How do you squeeze repeated characters with tr? | tr -s (--squeeze-repeats) replaces each run of a repeated listed character with a single occurrence. |
| cu-024 | 104 | lookup | What does ls -l show? | ls -l uses a long listing format: permissions, link count, owner, group, size, timestamp, and name per file. |
| cu-025 | 104 | lookup | How do you show hidden files with ls? | ls -a (--all) does not hide entries whose names start with a dot. |
| cu-026 | 124 | lookup | How do you copy directories recursively with cp? | cp -r (or -R, --recursive) copies directories and their contents recursively. |
| cu-027 | 124 | lookup | How do you preserve file attributes when copying? | cp -p preserves attributes such as mode, ownership, and timestamps; --preserve=LIST selects specific attributes. |
| cu-028 | 138 | lookup | How do you make mv prompt before overwriting? | mv -i (--interactive) prompts before overwriting an existing file. |
| cu-029 | 140 | lookup | How do you force rm to ignore nonexistent files? | rm -f (--force) ignores nonexistent files and arguments and never prompts. |
| cu-030 | 145 | lookup | How do you create a symbolic link with ln? | ln -s (--symbolic) creates symbolic links instead of hard links. |
| cu-031 | 149 | lookup | How do you create parent directories with mkdir? | mkdir -p (--parents) creates any missing parent directories and does not error if the directory already exists. |
| cu-032 | 165 | lookup | How do you show human-readable sizes with df? | df -h (--human-readable) prints sizes in powers of 1024 with unit suffixes (K, M, G). |
| cu-033 | 170 | lookup | How do you get a total-only summary with du? | du -s (--summarize) displays only a grand total for each argument instead of per-subdirectory sizes. |
| cu-034 | 218 | lookup | How do you print the date in UTC? | date -u (--utc/--universal) prints or sets the time in Coordinated Universal Time. |
| cu-035 | 222 | lookup | What does the nproc command print? | nproc prints the number of processing units available to the current process. |
| cu-036 | 254 | concept | What does the seq command print? | seq prints the numbers from FIRST to LAST, stepping by INCREMENT (default 1); options control width and separator. |
| cu-037 | 190 | lookup | How do you append to a file with tee? | tee -a (--append) appends to the given files instead of overwriting them. |
| cu-038 | 193 | concept | What does the basename command do? | basename strips the directory and an optional suffix from a path, printing just the final component. |
| cu-039 | 242 | concept | What does the timeout command do? | timeout runs a command with a time limit and sends it a signal (TERM by default) if it is still running after the given duration. |
| cu-040 | 182 | concept | What does the yes command do? | yes repeatedly prints a line consisting of its arguments (or 'y' if none) until it is killed. |
| cu-neg-01 | — | negative_cross_doc | How do you declare an associative array? | Not answerable from the coreutils manual. Associative arrays are a bash shell feature, not a coreutils utility. A correct response should decline and state the topic is not covered by this document. |
| cu-neg-02 | — | negative_cross_doc | What does the trap builtin do? | Not answerable from the coreutils manual. trap is a bash shell builtin, documented in the bash manual. A correct response should decline and state the topic is not covered by this document. |
| cu-neg-03 | — | negative_cross_doc | What is a pattern rule that uses %? | Not answerable from the coreutils manual. Pattern rules are a GNU make concept. A correct response should decline and state the topic is not covered by this document. |
| cu-neg-04 | — | negative_cross_doc | How do you run recipes in parallel with -j? | Not answerable from the coreutils manual. The -j parallelism flag is a GNU make option. A correct response should decline and state the topic is not covered by this document. |
| cu-neg-05 | — | negative_out_of_scope | What does the git rebase command do? | Not answerable from the coreutils manual. git is separate version-control software, not part of coreutils. A correct response should decline and state the topic is not covered by this document. |
| cu-neg-06 | — | negative_out_of_scope | How do you create a Kubernetes Deployment? | Not answerable from the coreutils manual. Kubernetes is unrelated to the coreutils manual. A correct response should decline and state the topic is not covered by this document. |
| cu-neg-07 | — | negative_out_of_scope | What does the SELECT statement do in SQL? | Not answerable from the coreutils manual. SQL is a database language outside all three manuals. A correct response should decline and state the topic is not covered by this document. |
| cu-neg-08 | — | negative_out_of_scope | How do you install a package with apt-get? | Not answerable from the coreutils manual. Package management with apt is outside the coreutils manual. A correct response should decline and state the topic is not covered by this document. |

## GNU Make Manual  (`corpus=make`  ≙ WMP PacifiCorp)
Source: `gnu-make-manual.pdf` · 229 pages · 42 in-scope + 8 negative = 50

| qid | pages | type | question | reference answer |
|---|---|---|---|---|
| mk-001 | 13 | concept | What does the make utility do? | make automatically determines which pieces of a large program need recompiling, based on file timestamps and the rules in a makefile, and issues the commands to rebuild them. |
| mk-002 | 35 | concept | What are the parts of a make rule? | A rule is: targets : prerequisites, followed by recipe lines each beginning with a TAB. The recipe runs when a target is older than a prerequisite. |
| mk-003 | 43 | concept | What is a phony target and how do you declare one? | A phony target is a name for a recipe rather than a real file; declaring it with .PHONY: name makes make always run its recipe and ignore any like-named file. |
| mk-004 | 143 | lookup | What does the automatic variable $@ represent? | $@ is the file name of the target of the rule being run. |
| mk-005 | 143 | lookup | What does the automatic variable $< represent? | $< is the name of the first prerequisite. |
| mk-006 | 143 | lookup | What does the automatic variable $^ represent? | $^ is the names of all prerequisites, space-separated, with duplicates removed. |
| mk-007 | 143 | lookup | What does the automatic variable $? represent? | $? is the names of all prerequisites that are newer than the target. |
| mk-008 | 144 | lookup | What does the automatic variable $* represent? | $* is the stem with which an implicit or pattern rule matched the target. |
| mk-009 | 141 | concept | What is a pattern rule? | A pattern rule looks like an ordinary rule but its target contains exactly one %; the % matches a stem, and the same stem substitutes into the prerequisites. |
| mk-010 | 37 | lookup | What does the wildcard function do? | $(wildcard pattern) expands to a space-separated list of existing file names matching the shell glob pattern. |
| mk-011 | 78 | concept | What is the difference between = and := variable assignment? | = creates a recursively expanded variable (its references are expanded each time it is used); := creates a simply expanded variable (expanded once, at definition time). |
| mk-012 | 79 | concept | How does a simply expanded variable behave? | A simply expanded variable, defined with := (or ::=), has its right-hand side expanded immediately at definition, so later changes to referenced variables do not affect it. |
| mk-013 | 81 | lookup | What does the ?= operator do? | ?= is conditional assignment: it sets the variable only if it is not already defined. |
| mk-014 | 86 | lookup | What does the += operator do? | += appends the given text (with a separating space) to a variable's existing value. |
| mk-015 | 63 | lookup | How do you run make jobs in parallel? | make -j [N] (--jobs) runs up to N recipes simultaneously; with no number it runs as many as possible. |
| mk-016 | 63 | lookup | What does the -k option do? | make -k (--keep-going) continues building other independent targets after a recipe fails, instead of stopping at the first error. |
| mk-017 | 123 | lookup | What does the -n option do? | make -n (--just-print/--dry-run) prints the recipes that would run without actually executing them. |
| mk-018 | 127 | lookup | What does the -B option do? | make -B (--always-make) treats all targets as out-of-date and rebuilds them unconditionally. |
| mk-019 | 25 | concept | What does the include directive do? | include reads one or more other makefiles at that point before continuing with the current one. |
| mk-020 | 69 | concept | What does the MAKEFLAGS variable do? | MAKEFLAGS carries the flags and options passed to make and is automatically communicated to sub-makes. |
| mk-021 | 39 | concept | What is VPATH used for? | VPATH is a list of directories make searches for prerequisites and targets not found in the current directory. |
| mk-022 | 40 | concept | What does the vpath directive do? | vpath pattern directories restricts the search to the listed directories only for file names matching pattern. |
| mk-023 | 104 | lookup | What does the patsubst function do? | $(patsubst pattern,replacement,text) finds whitespace-separated words in text matching pattern (with %) and replaces them with replacement. |
| mk-024 | 104 | lookup | What does the subst function do? | $(subst from,to,text) performs a plain textual replacement, changing every occurrence of from to to. |
| mk-025 | 105 | lookup | What does the filter function do? | $(filter pattern...,text) returns the words of text that match any of the patterns. |
| mk-026 | 106 | lookup | What does the filter-out function do? | $(filter-out pattern...,text) returns the words of text that do NOT match any of the patterns. |
| mk-027 | 106 | lookup | What does the sort function do? | $(sort list) sorts the words of list lexically and removes duplicates. |
| mk-028 | 111 | concept | What does the foreach function do? | $(foreach var,list,text) expands text once for each word in list, binding var to that word, and concatenates the results. |
| mk-029 | 109 | concept | What does the if function do? | $(if condition,then-part[,else-part]) expands to then-part when condition is non-empty, otherwise to else-part. |
| mk-030 | 111 | concept | What does the call function do? | $(call variable,param1,param2,...) invokes a variable as a macro, binding the params to $(1), $(2), etc. inside it. |
| mk-031 | 119 | concept | What does the shell function do? | $(shell command) runs command in a shell and returns its output, with newlines converted to spaces. |
| mk-032 | 107 | lookup | What does the dir function do? | $(dir names...) extracts the directory part (everything up to and including the last slash) of each name. |
| mk-033 | 107 | lookup | What does the notdir function do? | $(notdir names...) removes the directory part of each name, leaving the file component. |
| mk-034 | 97 | lookup | How do you write a conditional with ifeq? | ifeq (a,b) ... else ... endif runs the first branch when the two arguments are string-equal after expansion. |
| mk-035 | 99 | lookup | What does ifdef test? | ifdef takes a variable NAME (not a reference) and is true when that variable has a non-empty value. |
| mk-036 | 36 | concept | What is an order-only prerequisite? | Prerequisites listed after a \| are order-only: they must be built before the target, but their being newer does not by itself force the target to rebuild. |
| mk-037 | 92 | lookup | What does the .DEFAULT_GOAL variable do? | '.DEFAULT_GOAL' holds (and can set) the goal make builds when none is named on the command line; normally it is the first target. |
| mk-038 | 48 | concept | What does .DELETE_ON_ERROR do? | If '.DELETE_ON_ERROR' appears as a target, make deletes the target of any rule whose recipe fails, so no half-built file is left. |
| mk-039 | 59 | lookup | How do you silence a recipe line? | Prefix the recipe line with @; make then does not echo that line before running it. |
| mk-040 | 66 | lookup | How do you tell make to ignore an error in a recipe line? | Prefix the recipe line with -; make ignores a non-zero exit status from that line and keeps going. |
| mk-041 | 70 | concept | What does the export directive do? | export VAR puts a variable into the environment of the recipes make runs and passes it to sub-makes. |
| mk-042 | 118 | concept | What does the error function do? | $(error text) makes make print text as a fatal error and stop immediately when the function is expanded. |
| mk-neg-01 | — | negative_cross_doc | What does the $RANDOM variable expand to? | Not answerable from the make manual. $RANDOM is a bash shell variable, not a make feature. A correct response should decline and state the topic is not covered by this document. |
| mk-neg-02 | — | negative_cross_doc | How do you count the words in a file? | Not answerable from the make manual. Word counting is done by wc, a coreutils program. A correct response should decline and state the topic is not covered by this document. |
| mk-neg-03 | — | negative_cross_doc | What does the getopts builtin do? | Not answerable from the make manual. getopts is a bash shell builtin. A correct response should decline and state the topic is not covered by this document. |
| mk-neg-04 | — | negative_cross_doc | How do you create a symbolic link? | Not answerable from the make manual. Symbolic links are created with ln, a coreutils program. A correct response should decline and state the topic is not covered by this document. |
| mk-neg-05 | — | negative_out_of_scope | How do you deploy a smart contract on Ethereum? | Not answerable from the make manual. Blockchain deployment is unrelated to GNU make. A correct response should decline and state the topic is not covered by this document. |
| mk-neg-06 | — | negative_out_of_scope | How do you write a for loop in Python? | Not answerable from the make manual. Python syntax is outside the make manual. A correct response should decline and state the topic is not covered by this document. |
| mk-neg-07 | — | negative_out_of_scope | What is the boiling point of water at sea level? | Not answerable from the make manual. This is general trivia, not covered by the manual. A correct response should decline and state the topic is not covered by this document. |
| mk-neg-08 | — | negative_out_of_scope | What does the docker run command do? | Not answerable from the make manual. Docker is separate container software, not part of GNU make. A correct response should decline and state the topic is not covered by this document. |
