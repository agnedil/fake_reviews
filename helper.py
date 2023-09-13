import re
import ftfy
import matplotlib.pyplot as plt
import itertools
import numpy as np


def repair_text( s ):
    '''
        Clean up encoding, HTML leftovers, and other issues;
        full list of parameters - to enable flexibility when we need it.
        Examples:
                 "L&AMP;AMP;ATILDE;&AMP;AMP;SUP3;PEZ" ==> "LóPEZ"
                 "schÃ¶n" ==> "schön"
    '''
    return ftfy.fix_text(
                           s,
                           fix_encoding=True,
                           restore_byte_a0=True,
                           replace_lossy_sequences=True,
                           decode_inconsistent_utf8=True,
                           fix_c1_controls=True,
                           unescape_html=True,
                           remove_terminal_escapes=True,
                           fix_latin_ligatures=True,
                           fix_character_width=True,
                           uncurl_quotes=True,
                           fix_line_breaks=True,
                           fix_surrogates=True,
                           remove_control_chars=True,
                           normalization='NFC',
                           explain=False,
                        )


def clean_text( s,
                to_ascii=False,
              ):
    '''
        LIGHT GENERAL NON-DESTRUCTUVE TEXT CLEANING
    '''
    # edge case
    if not isinstance(s, str) or not s:
        return s

    # TODO: need list of such special characters that evade repair_text()
    for char in ['�', '•']:
        if char in s:
            s = s.replace(char, '')

    # fix text encoding
    s = repair_text( s )

    # convert to ascii
    if to_ascii:
        try:
            s = s.encode('ascii', 'ignore').decode()
        except:
            pass

    # replace line breaks
    s = s.replace('\n', ' ')    
    
    # remove multiple spaces
    s = re.sub('\s+', ' ', s)

    return s.strip()
    

# FUNCTION TO PRETTY PLOT THE CONFUSION MATRIX; USED LATER IN THE SCRIPT
def plot_confusion_matrix( cm, classes, fig_size=(7,7), fmt='.2f',    # 'd', '.2f'
                           title='Confusion matrix',
                           cmap=plt.cm.PuBu ):   # originally plt.cm.Blues; also good: BuPu,RdPu,PuRd,OrRd,Oranges
    """
    Plot the confusion matrix
    """
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
        
    plt.figure(figsize=fig_size)
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.05)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
        
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.tight_layout()
    plt.show()