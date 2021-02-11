import curses

def curses_init():
    ''' Initialize curses interface. '''
    stdscr = curses.initscr()
    stdscr.nodelay(True)
    curses.noecho()
    curses.cbreak()
    return stdscr

def curses_clean_up():
    ''' Clean up curses interface. '''
    curses.echo()
    curses.nocbreak()
    curses.endwin()

def progress_msg(stdscr, video_ordinal, word_count, video_name, num_videos):
    ''' Display preprocessing progress message. '''
    stdscr.addstr(1, 0, f'Processed {video_ordinal}/{num_videos} videos; {word_count} words.')
    stdscr.addstr(2, 0, f'Processing {video_name}...')
    stdscr.refresh()