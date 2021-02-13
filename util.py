import curses
import matplotlib.pyplot as plt

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

def plot_stat(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
 