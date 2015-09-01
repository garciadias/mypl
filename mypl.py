import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def stdpl(x, y, sx=None, sy=None, title='', xlabel='', ylabel='', color='r', marker='o',
		mksize=10, label='', save=None, xlim=[None,None], ylim=[None,None], ls='',
		width=35, height=25, xticks=25, yticks=25, legendsize=20):
	fig = plt.figure(figsize=(width*0.393701, height*0.393701))
	ax = fig.add_subplot(111)
	plt.tick_params(axis='x', which='both', labelsize=xticks)
	plt.tick_params(axis='y', which='both', labelsize=yticks)
	ax.set_title(title, fontsize=20)
	ax.set_xlabel(xlabel, fontsize=20)
	ax.set_ylabel(ylabel, fontsize=20)
	if(xlim[0]!=None and xlim[1]!=None):
		ax.set_xlim(xlim[0],xlim[1])
	if(ylim[0]!=None and ylim[1]!=None):
		ax.set_ylim(ylim[0],ylim[1])
	plt.errorbar(x, y, sx, sy, label=label, color=color, ls=ls, marker=marker)
	ax.legend(bbox_to_anchor=(1.1, 1.1), prop={'size':legendsize})
	plt.show()

if __name__=='__main__':
	import numpy as np
	x = range(100)
	y = range(100)
	stdpl(x,y)
