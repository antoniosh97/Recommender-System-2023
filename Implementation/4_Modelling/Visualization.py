import tensorboard
import webbrowser

def choose_runs():
    title = "  VISUALIZATION  "
    s = [
        "runs_StratifiedSamplingByUsers",
        "runs_StratifiedSamplingByItems",
        "runs_ProbabilitySampling",
    ]
    print("#"*len(title)+f"\n{title}\n"+"#"*len(title))
    print("Options: \n1. {sampling0}\n2. {sampling1}\n3. {sampling2}".format(
        sampling0=s[0],
        sampling1=s[1],
        sampling2=s[2]
    ))
    option = input("Choose an option: ")
    print(s[int(option)-1])
    return s[int(option)-1]

logs_base_dir = choose_runs()
tb = tensorboard.program.TensorBoard()
tb.configure(bind_all=True, logdir=f"4_Modelling/{logs_base_dir}")
url = tb.launch()
print(f"\nurl is {url}")
webbrowser.open_new_tab(url.replace("MSI","localhost"))