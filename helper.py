# Program 'helper.py' s komentarji.
# Ta program je namenjem risanju grafa rezultatov in za sam potek igre ni pomemben.
    
import matplotlib.pyplot as plt
from IPython import display
# Izposoja nekaterih knjižnic.
plt.ion()
# Zagon knjižnice.
def plot(scores, mean_scores):
    display.clear_output(wait=True) # Ponastavljanje.
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...') # Naslov.
    plt.xlabel('Number of games') # Število iger. Vhodna vrednost (x).
    plt.ylabel("Score") # Točke. Izhodna vrednost (t).
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    # Ostale vrstice so namenjene risanju, posodabljanju in splošnem delovanju grafa.
