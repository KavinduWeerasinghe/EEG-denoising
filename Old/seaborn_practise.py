import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
data=sns.load_dataset("tips")
ax=sns.boxplot(x=data["total_bill"])
ax=sns.swarmplot(x=data["total_bill"],color=".25")
ax.set(xlabel="Total Bill")
ax.set_title("Total Bill")
plt.show()
