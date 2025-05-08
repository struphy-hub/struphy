def add_likwid_parser(parser):
    # ---------------------------------------------------------------------
    parser_likwid = parser.add_argument_group("Likwid Options", "Arguments related to Likwid performance measurement")

    # Add Likwid-related arguments to the likwid group
    parser_likwid.add_argument(
        "--likwid",
        help="run with Likwid",
        action="store_true",
    )

    parser_likwid.add_argument(
        "-g",
        "--group",
        default="MEM_DP",
        type=str,
        help="likwid measurement group",
    )
    parser_likwid.add_argument(
        "--nperdomain",
        default=None, # Example: S:36 means 36 cores/socket
        type=str,
        help="Set the number of processes per node by giving an affinity domain and count",
    )

    parser_likwid.add_argument(
        "--stats",
        help="Print Likwid statistics",
        action="store_true",
    )

    parser_likwid.add_argument(
        "--marker",
        help="Activate Likwid marker API",
        action="store_true",
    )

    parser_likwid.add_argument(
        "--hpcmd_suspend",
        help="Suspend the HPCMD daemon",
        action="store_true",
    )

    parser_likwid.add_argument(
        "-lr",
        "--likwid-repetitions",
        type=int,
        help="number of repetitions of the same simulation",
        default=1,
    )
    # ---------------------------------------------------------------------------
