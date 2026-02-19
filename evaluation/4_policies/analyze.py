import pandas as pd
import argparse
import os

step_dict = {
    ("hl", "PPO", "control"): '20000000',
    ("hl", "DQN", "control"): '20000000',
    ("hl", "DQN", "vegan"): '2500000',
    ("hl", "PPO", "vegan"): '5000000',
    ("hl", "DQN", "vegetarian"): '10000000',
    ("hl", "PPO", "vegetarian"): '20000000',
    ("hl", "PPO", "earlybird"): '10000000',
    ("hl", "DQN", "earlybird"): '20000000',
    ("hl", "PPO", "contradiction"): '20000000',
    ("hl", "DQN", "contradiction"): '20000000',
    ("hl", "PPO", "penalty"): '20000000',
    ("hl", "DQN", "penalty"): '20000000',
    ("hl", "PPO", "solution"): '10000000',
    ("hl", "DQN", "solution"): '20000000',
    ("hl", "PPO", "solution-alt"): '10000000',
    ("hl", "DQN", "solution-alt"): '20000000',
    ("img", "DQN", "control"): '20000000',
    ("img", "DQN", "vegan"): '20000000',
    ("img", "DQN", "vegetarian"): '20000000',
    ("img", "DQN", "earlybird"): '20000000',

}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", help="PPO or DQN")
    parser.add_argument("--data", help="hl or images")
    args = parser.parse_args()
    dt = args.data
    algo = args.algo

    if dt == 'hl':
        print("---------------------------CONTROL STATS-------------------------")

        df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"/"+dt+"_control_" + algo + "_" + step_dict[(dt, algo, "control")]+"/"+dt+"_control_" + algo + "_" + step_dict[(dt, algo, "control")] + "_0"+"{:02d}".format(0)+".csv", header=0, index_col=False)
        for i in range(1,20):
            temp = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"/"+dt+"_control_" + algo + "_" + step_dict[(dt, algo, "control")]+"/"+dt+"_control_" + algo + "_" + step_dict[(dt, algo, "control")] + "_0"+"{:02d}".format(i)+".csv", header=0, index_col=False)
            df = pd.concat([df, temp], ignore_index=True)
        score = df["Score"].mean()
        print("Average score: ", score)
        print("% of max score: ", score / 1795.0)
        print("Win %", df["Win/Lose"].value_counts().get("win", 0) * 100.0 / len(df))

        print("----------Vegan------------")
        viol_ct = df["VegetarianBlue"].sum() + df["VegetarianOrange"].sum()
        print("Total Vegan violations: ", viol_ct)
        print("Average violations per game: ", viol_ct / len(df))

        print("----------Vegetarian------------")
        print("Blue ghosts eaten: ", df["Blue Eaten"].sum())
        print("% ghosts eaten (blue): ", df["Blue Eaten"].sum() / viol_ct)
        print("Orange ghosts eaten: ", df["Orange Eaten"].sum())
        print("% ghosts eaten (orange): ", df["Orange Eaten"].sum() / viol_ct)
        viol_ct = df["VegetarianOrange"].sum()
        print("Total Vegatarian violations: ", viol_ct)
        print("Average violations per game: ", viol_ct / len(df))

        print("----------Penalty------------") #FIX
        viol_ct = df["Penalty1(total)"].sum()
        print("Total violations (Penalty): ", viol_ct)
        print("Average violations per game (Penalty): ", viol_ct / len(df))
        viol_ct = df["CTD"].sum()
        print("CTD violations (Penalty): ", viol_ct)
        print("Average CTD violations per game (Penalty): ", viol_ct / len(df))

        print("----------EarlyBird------------")
        viol_ct = df["EarlyBird"].sum()
        print("Total violations (EarlyBird): ", viol_ct)
        print("Average violations per game: ", viol_ct / len(df))

        print("-")
        print("-")

        print("------------------------EVALUATION STATS-------------------------")

        print("----------Vegan------------")
        #replace DQN with algo
        df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"/"+dt+"_vegan_" + algo + "_" + step_dict[(dt, algo, "vegan")]+"/"+dt+"_vegan_" + algo + "_" + step_dict[(dt, algo, "vegan")] + "_0"+"{:02d}".format(0)+".csv", header=0, index_col=False)
        for i in range(1,20):
            temp = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"/"+dt+"_vegan_" + algo + "_" + step_dict[(dt, algo, "vegan")]+"/"+dt+"_vegan_" + algo + "_" + step_dict[(dt, algo, "vegan")] + "_0"+"{:02d}".format(i)+".csv", header=0, index_col=False)
            df = pd.concat([df, temp], ignore_index=True)
        score = df["Score"].mean()
        print("Average score: ", score)
        print(" % of max score: ", score / 995.0)
        print("Win %", df["Win/Lose"].value_counts().get("win", 0) * 100.0 / len(df))

        viol_ct = df["Blue Eaten"].sum() + df["Orange Eaten"].sum()
        print("Total Vegan violations: ", viol_ct)
        print("Average violations per game: ", viol_ct / len(df))

        print("----------Vegetarian------------")
        #replace DQN with algo
        df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"/"+dt+"_vegetarian_" + algo + "_" + step_dict[(dt, algo, "vegetarian")]+"/"+dt+"_vegetarian_" + algo + "_" + step_dict[(dt, algo, "vegetarian")] + "_0"+"{:02d}".format(0)+".csv", header=0, index_col=False)
        for i in range(1,20):
            temp = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"/"+dt+"_vegetarian_" + algo + "_" + step_dict[(dt, algo, "vegetarian")]+"/"+dt+"_vegetarian_" + algo + "_" + step_dict[(dt, algo, "vegetarian")] + "_0"+"{:02d}".format(i)+".csv", header=0, index_col=False)
            df = pd.concat([df, temp], ignore_index=True)
        score = df["Score"].mean()
        print("Average score: ", score)
        print(" % of max score: ", score / 1395.0)
        print("Win %", df["Win/Lose"].value_counts().get("win", 0) * 100.0 / len(df))

        viol_ct = df["VegetarianOrange"].sum()
        print("Total violations (orange ghost): ", viol_ct)
        print("Average violations per game: ", viol_ct / len(df))

        perm_ct = df["Blue Eaten"].sum()
        print("Total permissible ghosts eaten: ", perm_ct)
        print("Average permissible ghosts eaten per game: ", perm_ct / len(df))

        print("----------Penalty------------")
        df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"/"+dt+"_penalty1_" + algo + "_" + step_dict[(dt, algo, "penalty")]+"/"+dt+"_penalty1_" + algo + "_" + step_dict[(dt, algo, "penalty")] + "_0"+"{:02d}".format(0)+".csv", header=0, index_col=False)
        for i in range(1,20):
            temp = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"/"+dt+"_penalty1_" + algo + "_" + step_dict[(dt, algo, "penalty")]+"/"+dt+"_penalty1_" + algo + "_" + step_dict[(dt, algo, "penalty")] + "_0"+"{:02d}".format(0)+".csv", header=0, index_col=False)
            df = pd.concat([df, temp], ignore_index=True)
        score = df["Score"].mean()
        print("Average score: ", score)
        print(" % of max score: ", score / 995.0)
        print("Win %", df["Win/Lose"].value_counts().get("win", 0) * 100.0 / len(df))

        viol_ct = df["Vegan"].sum()
        print("Vegan Violations: ", viol_ct)
        print("Average Vegan Violations per game: ", viol_ct / len(df))

        viol_ct = df["EarlyBird"].sum()
        print("EarlyBird violations: ", viol_ct)
        print("Average EarlyBird violations per game: ", viol_ct / len(df))

        viol_ct = df["CTD"].sum()
        print("CTD violations (Penalty): ", viol_ct)
        print("Average CTD violations per game (Penalty): ", viol_ct / len(df))


        print("----------EarlyBird------------")
        #turn PPO to algo
        df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"/"+dt+"_earlybird_" + algo + "_" + step_dict[(dt, algo, "earlybird")]+"/"+dt+"_earlybird_" + algo + "_" + step_dict[(dt, algo, "earlybird")] + "_0"+"{:02d}".format(0)+".csv", header=0, index_col=False)
        for i in range(1,20):
            temp = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"/"+dt+"_earlybird_" + algo + "_" + step_dict[(dt, algo, "earlybird")]+"/"+dt+"_earlybird_" + algo + "_" + step_dict[(dt, algo, "earlybird")] + "_0"+"{:02d}".format(i)+".csv", header=0, index_col=False)
            df = pd.concat([df, temp], ignore_index=True)
        score = df["Score"].mean()
        print("Average score: ", score)
        print(" % of max score: ", score / 1795.0)
        print("Win %", df["Win/Lose"].value_counts().get("win", 0) * 100.0 / len(df))

        viol_ct = df["EarlyBird"].sum()
        print("Total violations (EarlyBird): ", viol_ct)
        print("Average violations per game: ", viol_ct / len(df))

        print("----------Contradiction------------")
        #turn PPO to algo
        df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"/"+dt+"_contradiction_" + algo + "_20000000/"+dt+"_contradiction_" + algo + "_20000000_0"+"{:02d}".format(0)+".csv", header=0, index_col=False)
        for i in range(1,20):
            temp = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"/"+dt+"_contradiction_" + algo + "_20000000/"+dt+"_contradiction_" + algo + "_20000000_0"+"{:02d}".format(i)+".csv", header=0, index_col=False)
            df = pd.concat([df, temp], ignore_index=True)
        score = df["Score"].mean()
        print("Average score: ", score)
        print(" % of max score: ", score / 1595.0)
        print("Win %", df["Win/Lose"].value_counts().get("win", 0) * 100.0 / len(df))

        viol_eb = df["EarlyBird"].sum()
        print("Total violations (EarlyBird): ", viol_eb)
        print("Average violations per game: ", viol_eb / len(df))

        viol_bg = df["VegetarianBlue"].sum()
        viol_og = df["VegetarianOrange"].sum()
        print("Total violations (blue ghost): ", viol_bg)
        print("Average violations per game: ", viol_bg / len(df))
        print("% ghosts eaten (blue): ", viol_bg / (viol_bg + viol_og))
        print("Total violations (orange ghost): ", viol_og)
        print("Average violations per game: ", viol_og / len(df))
        print("% ghosts eaten (orange): ", viol_og / (viol_bg + viol_og))

        print(
        "# games with more than one ghost eaten: ",
         ((df["VegetarianBlue"].map(float) + df["VegetarianOrange"].map(float)) > 1.0).sum(),
        )

        eb_viol_df = df[df["EarlyBird"].map(float) > 0.0]

        print(
        "# games where ghost is eaten but EarlyBird is violated: ",
        (
            (eb_viol_df["VegetarianBlue"].map(float) + eb_viol_df["VegetarianOrange"].map(float))
            > 0.0
        ).sum(),
        )


        print(
        "% games with more than one ghost eaten: ",
        ((df["VegetarianBlue"].map(float) + df["VegetarianOrange"].map(float)) > 1.0).sum()*100.0/len(df),
        )

        eb_viol_df = df[df["EarlyBird"].map(float) > 0.0]

        print(
        "% games where ghost is eaten but EarlyBird is violated: ",
        (
            (eb_viol_df["VegetarianBlue"].map(float) + eb_viol_df["VegetarianOrange"].map(float))
            > 0.0
        ).sum()*100.0/len(df),
        )

        print("----------Solution------------")
        df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"/"+dt+"_solution_" + algo + "_" + step_dict[(dt, algo, "solution")]+"/"+dt+"_solution_" + algo + "_" + step_dict[(dt, algo, "solution")] + "_0"+"{:02d}".format(0)+".csv", header=0, index_col=False)
        for i in range(1,20):
            temp = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"/"+dt+"_solution_" + algo + "_" + step_dict[(dt, algo, "solution")]+"/"+dt+"_solution_" + algo + "_" + step_dict[(dt, algo, "solution")] + "_0"+"{:02d}".format(0)+".csv", header=0, index_col=False)
            df = pd.concat([df, temp], ignore_index=True)
        score = df["Score"].mean()
        print("Average score: ", score)
        print(" % of max score: ", score / 1595.0)
        print("Win %", df["Win/Lose"].value_counts().get("win", 0) * 100.0 / len(df))

        viol_eb = df["EarlyBird"].sum()
        print("Total violations (EarlyBird): ", viol_eb)
        print("Average violations per game: ", viol_eb / len(df))

        eaten_bg = df["Blue Eaten"].sum()
        viol_og = df["VegetarianOrange"].sum()
        print("Blue ghosts eaten: ", eaten_bg)
        print("Average blue ghosts eaten per game: ", eaten_bg / len(df))
        print("% ghosts eaten (blue): ", eaten_bg / (eaten_bg + viol_og))

        print("Total violations (orange ghost): ", viol_og)
        print("Average violations per game: ", viol_og / len(df))
        print("% ghosts eaten (orange): ", viol_og / (eaten_bg + viol_og))


        print("----------Solution-ALT------------")
        df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"/"+dt+"_solution-altweights_" + algo + "_" + step_dict[(dt, algo, "solution-alt")]+"/"+dt+"_solution-altweights_" + algo + "_" + step_dict[(dt, algo, "solution-alt")] + "_0"+"{:02d}".format(0)+".csv", header=0, index_col=False)
        for i in range(1,20):
            temp = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"/"+dt+"_solution-altweights_" + algo + "_" + step_dict[(dt, algo, "solution-alt")]+"/"+dt+"_solution-altweights_" + algo + "_" + step_dict[(dt, algo, "solution-alt")] + "_0"+"{:02d}".format(0)+".csv", header=0, index_col=False)
            df = pd.concat([df, temp], ignore_index=True)
        score = df["Score"].mean()
        print("Average score: ", score)
        print(" % of max score: ", score / 1595.0)
        print("Win %", df["Win/Lose"].value_counts().get("win", 0) * 100.0 / len(df))

        viol_eb = df["EarlyBird"].sum()
        print("Total violations (EarlyBird): ", viol_eb)
        print("Average violations per game: ", viol_eb / len(df))

        eaten_bg = df["Blue Eaten"].sum()
        viol_og = df["VegetarianOrange"].sum()
        print("Blue ghosts eaten: ", eaten_bg)
        print("Average blue ghosts eaten per game: ", eaten_bg / len(df))
        print("% ghosts eaten (blue): ", eaten_bg / (eaten_bg + viol_og))

        print("Total violations (orange ghost): ", viol_og)
        print("Average violations per game: ", viol_og / len(df))
        print("% ghosts eaten (orange): ", viol_og / (eaten_bg + viol_og))














    
    if dt == 'img':
        print("---------------------------CONTROL STATS-------------------------")

        df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"/"+dt+"_control_" + algo + "_" + step_dict[(dt, algo, "control")]+"/"+dt+"_control_" + algo + "_" + step_dict[(dt, algo, "control")] + "_f3_0"+"{:02d}".format(0)+".csv", header=0, index_col=False)
        for i in range(1,5):
            temp = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"/"+dt+"_control_" + algo + "_" + step_dict[(dt, algo, "control")]+"/"+dt+"_control_" + algo + "_" + step_dict[(dt, algo, "control")] + "_f3_0"+"{:02d}".format(i)+".csv", header=0, index_col=False)
            df = pd.concat([df, temp], ignore_index=True)
        score = df["Score"].mean()
        print("Average score: ", score)
        print("% of max score: ", score / 1795.0)
        print("Win %", df["Win/Lose"].value_counts().get("win", 0) * 100.0 / len(df))

        print("----------Vegan------------")
        viol_ct = df["VegetarianBlue"].sum() + df["VegetarianOrange"].sum()
        print("Total Vegan violations: ", viol_ct)
        print("Average violations per game: ", viol_ct / len(df))

        print("----------Vegetarian------------")
        print("Blue ghosts eaten: ", df["Blue Eaten"].sum())
        print("% ghosts eaten (blue): ", df["Blue Eaten"].sum() / viol_ct)
        print("Orange ghosts eaten: ", df["Orange Eaten"].sum())
        print("% ghosts eaten (orange): ", df["Orange Eaten"].sum() / viol_ct)
        viol_ct = df["VegetarianOrange"].sum()
        print("Total Vegatarian violations: ", viol_ct)
        print("Average violations per game: ", viol_ct / len(df))

        print("----------Penalty------------") #FIX
        viol_ct = df["Penalty1(total)"].sum()
        print("Total violations (Penalty): ", viol_ct)
        print("Average violations per game (Penalty): ", viol_ct / len(df))
        viol_ct = df["CTD"].sum()
        print("CTD violations (Penalty): ", viol_ct)
        print("Average CTD violations per game (Penalty): ", viol_ct / len(df))

        print("----------EarlyBird------------")
        viol_ct = df["EarlyBird"].sum()
        print("Total violations (EarlyBird): ", viol_ct)
        print("Average violations per game: ", viol_ct / len(df))

        print("-")
        print("-")

        print("------------------------EVALUATION STATS-------------------------")

        print("----------Vegan------------")
        #replace DQN with algo
        df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"/"+dt+"_vegan_" + algo + "_" + step_dict[(dt, algo, "vegan")]+"/"+dt+"_vegan_" + algo + "_" + step_dict[(dt, algo, "vegan")] + "_f3_0"+"{:02d}".format(0)+".csv", header=0, index_col=False)
        for i in range(1,5):
            temp = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"/"+dt+"_vegan_" + algo + "_" + step_dict[(dt, algo, "vegan")]+"/"+dt+"_vegan_" + algo + "_" + step_dict[(dt, algo, "vegan")] + "_f3_0"+"{:02d}".format(i)+".csv", header=0, index_col=False)
            df = pd.concat([df, temp], ignore_index=True)
        score = df["Score"].mean()
        print("Average score: ", score)
        print(" % of max score: ", score / 995.0)
        print("Win %", df["Win/Lose"].value_counts().get("win", 0) * 100.0 / len(df))

        viol_ct = df["Blue Eaten"].sum() + df["Orange Eaten"].sum()
        print("Total Vegan violations: ", viol_ct)
        print("Average violations per game: ", viol_ct / len(df))

        print("----------Vegetarian------------")
        #replace DQN with algo
        df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"/"+dt+"_vegetarian_" + algo + "_" + step_dict[(dt, algo, "vegetarian")]+"/"+dt+"_vegetarian_" + algo + "_" + step_dict[(dt, algo, "vegetarian")] + "_f3_0"+"{:02d}".format(0)+".csv", header=0, index_col=False)
        for i in range(1,5):
            temp = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"/"+dt+"_vegetarian_" + algo + "_" + step_dict[(dt, algo, "vegetarian")]+"/"+dt+"_vegetarian_" + algo + "_" + step_dict[(dt, algo, "vegetarian")] + "_f3_0"+"{:02d}".format(i)+".csv", header=0, index_col=False)
            df = pd.concat([df, temp], ignore_index=True)
        score = df["Score"].mean()
        print("Average score: ", score)
        print(" % of max score: ", score / 1395.0)
        print("Win %", df["Win/Lose"].value_counts().get("win", 0) * 100.0 / len(df))

        viol_ct = df["VegetarianOrange"].sum()
        print("Total violations (orange ghost): ", viol_ct)
        print("Average violations per game: ", viol_ct / len(df))

        perm_ct = df["Blue Eaten"].sum()
        print("Total permissible ghosts eaten: ", perm_ct)
        print("Average permissible ghosts eaten per game: ", perm_ct / len(df))


        print("----------EarlyBird------------")
        #turn PPO to algo
        df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"/"+dt+"_earlybird_" + algo + "_" + step_dict[(dt, algo, "earlybird")]+"/"+dt+"_earlybird_" + algo + "_" + step_dict[(dt, algo, "earlybird")] + "_f3_0"+"{:02d}".format(0)+".csv", header=0, index_col=False)
        for i in range(1,5):
            temp = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"/"+dt+"_earlybird_" + algo + "_" + step_dict[(dt, algo, "earlybird")]+"/"+dt+"_earlybird_" + algo + "_" + step_dict[(dt, algo, "earlybird")] + "_f3_0"+"{:02d}".format(i)+".csv", header=0, index_col=False)
            df = pd.concat([df, temp], ignore_index=True)
        score = df["Score"].mean()
        print("Average score: ", score)
        print(" % of max score: ", score / 1795.0)
        print("Win %", df["Win/Lose"].value_counts().get("win", 0) * 100.0 / len(df))

        viol_ct = df["EarlyBird"].sum()
        print("Total violations (EarlyBird): ", viol_ct)
        print("Average violations per game: ", viol_ct / len(df))



