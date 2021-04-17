    import matplotlib.pyplot as plt
    from IPython import display

    plt.ion()
    #v tem podprogramu je program ki izpiše graf
    def plot(scores, mean_scores):
        display.clear_output(wait=True)#pobriše
        display.display(plt.gcf())
        plt.clf()
        plt.title('Training...')#napiše naslov trening
        plt.xlabel('Number of Games')#napiše na eno stran Number of Games
        plt.ylabel("Score")#na drugo stran napiše Score
        plt.plot(scores)
        plt.plot(mean_scores)
        plt.ylim(ymin=0)
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))#tukaj izbiše graf
        plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))#tukaj tudi izpiše graf
