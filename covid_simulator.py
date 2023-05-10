from matplotlib.animation import ArtistAnimation
from matplotlib import pyplot, colors
from matplotlib import animation
import PySimpleGUI as sg
import numpy as np
import random
import copy
import csv
import os

# TODO: Train simple model to predict best move for agent
# TODO: 3 dimensional field is stupidly ineffective, think of something better
# TODO: Train a model for getting best hyperparameters
# TODO: Implement batching for simulation

# input part
file_path = os.path.dirname(__file__)
sg.user_settings_filename(filename='saved_settings.json', path=file_path)
sg.theme('DarkAmber')  # Add a touch of color
layout = [[sg.Text('Input desired parameters')],
          [sg.Text(
              "The simulation consists of a number of time-steps, in which each agent moves randomly by a distance\n"
              "defined by its mobility parameter V. The distance every agent moves in one time-step\n "
              "is chosen randomly in the range [0, 2×V].\n"
              "Every agent can be in one of three states: susceptible, ill, or immune (either vaccinated or after "
              "recovery). \n "
              "There is one more state possible: dead, which is equivalent to removing the agent from the simulation\n"
              "(however, the death count should be tracked).\n"
              "If an susceptible agent is within the distance d (for COVID19 it is estimated to 1–2m) from the ill "
              "agent,\n "
              "there is a p probability of changing the state to ill.\n"
              "After incubation time t1 (measured in time steps) from the infection, the mobility v of the ill agent "
              "is reduced to 0\n "
              "(the person has developed symptoms and has been isolated).\n"
              "After time t2 from the infection, the state of ill agent is changed to immune or dead.\n"
              "The mortality rate m is the probability of death.\n"
              "Please note that the maximum size of the simulation should not exceed 10000\n"),
              sg.Text("load or save settings (or do neither)"), sg.Button("load", key="load"), sg.Button("save", key="save")],
          [sg.Checkbox("total cases over time graph", key="-total_cases_graph-"), sg.Push(),
           sg.Text("mobility parameter (mobility)"), sg.Input(key="-mobility-")],
          [sg.Checkbox("deaths over time graph", key="-death_graph-"), sg.Push(),
           sg.Text("distance (distance)"), sg.Input(key="-distance-")],
          [sg.Checkbox("ill over time graph", key="-ill_graph-"), sg.Push(),
           sg.Text("incubation time (incubation)"), sg.Input(key="-incubation-")],
          [sg.Checkbox("susceptible over time graph", key="-susceptible_graph-"), sg.Push(),
           sg.Text("duration of illness after incubation (duration)"), sg.Input(key="-duration-")],
          [sg.Checkbox("immune over time graph", key="-immune_graph-"), sg.Push(),
           sg.Text("probability of spreading the disease (probability)"), sg.Input(key="-probability-")],
          [sg.Push(), sg.Text("mortality rate (mortality)"), sg.Input(key="-mortality-")],
          [sg.Checkbox("R0 over time graph", key="-R0_graph-"), sg.Push(),
           sg.Text("simulation size (size)"), sg.Input(key="-size-")],
          [sg.Push(), sg.Text("simulation length (simulation_length)"), sg.Input(key="-simulation_length-")],
          [sg.Push(), sg.Text("number of susceptible agents (susceptible)"), sg.Input(key="-susceptible-")],
          [sg.Checkbox("save ml data", key="-save_data-"), sg.Push(), sg.Text("number of ill agents (ill)"), sg.Input(key="-ill-")],
          [sg.Checkbox("machine learning", key="-ml-"), sg.Push(), sg.Text("number of immune agents (immune)"), sg.Input(key="-immune-")],
          [sg.Checkbox("save animation", key="-save_animation-"), sg.Push(),
           sg.Submit(), sg.Cancel()]]

window = sg.Window('Simulation settings', layout, element_justification='c', keep_on_top=True, grab_anywhere=True)
while True:
    event, values = window.read()

    # I am sorry for this, can't really make this less ugly
    if event == 'load':
        window.Element('-mobility-').Update(sg.user_settings_get_entry('-mobility-'))
        window.Element('-distance-').Update(sg.user_settings_get_entry('-distance-'))
        window.Element('-incubation-').Update(sg.user_settings_get_entry('-incubation-'))
        window.Element('-duration-').Update(sg.user_settings_get_entry('-duration-'))
        window.Element('-probability-').Update(sg.user_settings_get_entry('-probability-'))
        window.Element('-mortality-').Update(sg.user_settings_get_entry('-mortality-'))
        window.Element('-size-').Update(sg.user_settings_get_entry('-size-'))
        window.Element('-simulation_length-').Update(sg.user_settings_get_entry('-simulation_length-'))
        window.Element('-susceptible-').Update(sg.user_settings_get_entry('-susceptible-'))
        window.Element('-ill-').Update(sg.user_settings_get_entry('-ill-'))
        window.Element('-immune-').Update(sg.user_settings_get_entry('-immune-'))

    elif event == 'save':
        sg.user_settings_set_entry('mobility', int(values['-mobility-']))
        sg.user_settings_set_entry('-distance-', int(values['-distance-']))
        sg.user_settings_set_entry('-incubation-', int(values['-incubation-']))
        sg.user_settings_set_entry('-duration-', int(values['-duration-']) + int(values['-incubation-']))
        sg.user_settings_set_entry('-probability-', float(values['-probability-']))
        sg.user_settings_set_entry('-mortality-', float(values['-mortality-']))
        sg.user_settings_set_entry('-size-', int(values['-size-']))
        sg.user_settings_set_entry('-simulation_length-', int(values['-simulation_length-']))
        sg.user_settings_set_entry('-susceptible-', int(values['-susceptible-']))
        sg.user_settings_set_entry('-ill-', int(values['-ill-']))
        sg.user_settings_set_entry('-immune-', int(values['-immune-']))
        values['load_save_none'] = 'neither'

    elif event == sg.WIN_CLOSED or event == 'Cancel' or event == 'Submit':
        break

mobility, distance, incubation, probability = int(values["-mobility-"]), int(values['-distance-']), int(values['-incubation-']), float(values['-probability-'])
duration = int(values['-duration-']) + incubation
mortality, size, simulation_length = float(values['-mortality-']), int(values['-size-']), int(values['-simulation_length-'])
susceptible, ill, immune = int(values['-susceptible-']), int(values['-ill-']), int(values['-immune-'])

window.close()


# Simulation part
class Analysis:

    class Agent:
        def __init__(self, x, y, group):
            self.x = x
            self.y = y
            self.group = group
            self.mobility = random.choice(range(1, 2 * mobility))
            self.duration = 0
            self.incubation = 0
            self.id = random.choice(range(0, (size * size)))

    def agent_creator(self,group_size, group, group_list, field, n, m):
        for j in range(0, group_size):
            while True:
                x = random.choice(range(0, n))
                y = random.choice(range(0, m))
                if field[x][y] is None:
                    field[x][y] = self.Agent(x=x, y=y, group=group)
                    group_list.append(field[x][y])
                    break

    def is_id_unique(self, agent, agent_list):
        for other_agent in agent_list:
            if agent.id == other_agent.id:
                agent.id = random.choice(range(0, (size * size)))

    def create_groups(self):
        field = np.array(([[None for _ in range(0, size)] for _ in range(0, size)]))
        susceptible_list, immune_list, ill_list, dead_list = [], [], [], []
        self.agent_creator(susceptible, "susceptible", susceptible_list, field, size, size)
        self.agent_creator(ill, "ill", ill_list, field, size, size)
        self.agent_creator(immune, "immune",  immune_list, field, size, size)
        self.all_agents = susceptible_list + immune_list + ill_list
        for agent in self.all_agents:
            self.is_id_unique(agent, self.all_agents)
        return susceptible_list, immune_list, ill_list, dead_list, field

    def create_original_agent_range(self, agent):
        x_range = list(range((agent.x - mobility), (agent.x + mobility)))
        y_range = list(range((agent.y - mobility), (agent.y + mobility)))
        return x_range, y_range


    def move_agent(self, agent):
        new_x = (agent.x + random.choice(range(-agent.mobility, agent.mobility + 1))) % len(field)
        new_y = (agent.y + random.choice(range(-agent.mobility, agent.mobility + 1))) % len(field[0])
        if values['-ml-']:
            x_range, y_range = self.create_original_agent_range(agent)
            min_x, max_x = x_range[0], x_range[-1]
            min_y, max_y = y_range[0], y_range[-1]
            if min_x <= new_x <= max_x and min_y <= new_y <= max_y:
                field[agent.x][agent.y] = None
                field[new_x][new_y] = agent
                agent.x = new_x
                agent.y = new_y
            else:
                pass
        else:
            field[agent.x][agent.y] = None
            if field[new_x][new_y] is not None:
                field[agent.x][agent.y] = agent
            else:
                agent.x = new_x
                agent.y = new_y
                field[agent.x][agent.y] = agent

    def agent_sight(self, agent):
        self.FieldState = []
        for i in range(agent.x - mobility, agent.x + mobility):
            for j in range(agent.y - mobility, agent.y + mobility):
                self.FieldState.append(["Field:", field[i % len(field)][j % len(field[0])] is not None,
                                        "Group:",
                                        field[i % len(field)][j % len(field[0])].group if field[i % len(field)][
                                                                                              j % len(field[
                                                                                                          0])] is not None else None,
                                        "Position:", i % len(field), j % len(field[0])])
        return self.FieldState

    def save_sight(self, agent, timestep):
        self.agent_sight(agent)
        with open('sight.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([agent.id, timestep, agent.group, [agent.x, agent.y], agent.mobility, self.FieldState])

    @staticmethod
    def scan_for_ill(agent):
        for i in range((agent.x - distance) -1, (agent.x + distance) -1):
            for j in range((agent.y - distance) -1, (agent.y + distance) -1):
                if size < i:
                    i = i - size
                if size < j:
                    j = j - size
                if field[i][j] is None:
                    continue
                if field[i][j].group == "ill":
                    return True

    def infect_agent(self, agent):
        if agent.group != 'susceptible':
            return None
        if not self.scan_for_ill(agent):
            return None
        if random.choice(range(0, 100)) < probability:
            agent.group = "ill"
            susceptible_list.remove(agent)
            ill_list.append(agent)

    @staticmethod
    def update_illness_state(agent):
        if agent.group == "ill":
            agent.duration += 1
            agent.incubation += 1

    @staticmethod
    def change_agent_state(agent):
        if agent.group != "ill":
            return None
        if agent.duration <= duration:
            return None
        if random.choice(range(0, 100)) < mortality:
            agent.group = "dead"
            ill_list.remove(agent)
            dead_list.append(agent)
        else:
            agent.group = "immune"
            ill_list.remove(agent)
            immune_list.append(agent)



    @staticmethod
    def quarantine(agent):
        if agent.group == "ill":
            if agent.incubation > incubation:
                agent.mobility = 0
        elif agent.group == "immune":
            agent.mobility = mobility

    @staticmethod
    def remove_dead_agents(agent):
        if agent.group == "dead":
            field[agent.x][agent.y] = None


def main(save_data=False):
    all_agents = [agent for agent in susceptible_list + immune_list + ill_list]
    for timestep in range(simulation_length):
        for agent in all_agents:
            if agent.group != "dead":
                analysis.move_agent(agent)
                analysis.update_illness_state(agent)
                analysis.change_agent_state(agent)
                analysis.quarantine(agent)
                if save_data is True:
                    analysis.save_sight(agent, timestep)
                analysis.infect_agent(agent)
            analysis.remove_dead_agents(agent)
        print(f"timestep: {timestep}")
        current_field = [timestep, field, dead_list, susceptible_list, immune_list, ill_list]
        yield current_field


# visualization part
class Visualisation:
    population = np.zeros((size, size))

    def __init__(self):
        self.dead_graph_list = []
        self.immune_graph_list = []
        self.susceptible_graph_list = []
        self.ill_graph_list = []
        self.total_graph_list = []
        self.R0_population_density_list = []
        self.actuall_R0_list = []
        self.ims = []

    def populate(self):
        for i in range(0, size):
            for j in range(0, size):
                if field[i][j] is None:
                    self.population[i][j] = 0.0
                elif field[i][j].group == "susceptible":
                    self.population[i][j] = 1.0
                elif field[i][j].group == "ill":
                    self.population[i][j] = 2.0
                elif field[i][j].group == "immune":
                    self.population[i][j] = 3.0

    def average_infected(self):
        ill_in_timestep_list = []
        for element in self.ill_graph_list:
            time = element[0]
            ill = element[1]
            previous_ill = self.ill_graph_list[time - 1][1] if time - 1 >= 0 else 0
            ill_in_timestep_list.append(ill - previous_ill)
            average_infected = sum(ill_in_timestep_list) / len(ill_in_timestep_list)
            yield (time, average_infected)

    def animate_plot(self):
        # i actually hate matplotlib
        # Ceterum censeo Matplotlib esse delendam
        fig = pyplot.figure(figsize=(12, 12))
        pyplot.title("COVID-19 simulation\n"
                     "Yellow = Ill, Blue = Susceptible, Green = Immune", fontsize=24)
        pyplot.xlabel("x coordinates", fontsize=20)
        pyplot.ylabel("y coordinates", fontsize=20)
        pyplot.xticks(fontsize=16)
        pyplot.yticks(fontsize=16)
        colormap = colors.ListedColormap(["black", "blue", "yellow", "green"])
        for timestep, field, dead_list, susceptible_list, immune_list, ill_list in main(save_data=values["-save_data-"]):
            self.populate()
            self.dead_graph_list.append((timestep, len(dead_list)))

            self.immune_graph_list.append((timestep, len(immune_list)))
            self.susceptible_graph_list.append((timestep, len(susceptible_list)))
            self.ill_graph_list.append((timestep, len(ill_list)))
            im = pyplot.imshow(self.population ,cmap=colormap, animated=True)
            self.ims.append([im])
        anim = ArtistAnimation(fig, self.ims, interval=300, repeat_delay=1500)
        if values["-save_animation-"]:
            writervideo = animation.FFMpegWriter()
            anim.save(f'{size}x{size}.gif', writer=writervideo)
        pyplot.show()

    def total_cases_graph(self):
        fig_total = pyplot.figure(figsize=(12, 12))
        for ill, dead, immune, in zip(self.ill_graph_list, self.dead_graph_list, self.immune_graph_list):
            self.total_graph_list.append((ill[0], ill[1] + dead[1] + immune[1]))
        pyplot.plot(*zip(*self.total_graph_list), label="Total cases")
        pyplot.title("Total cases over time", fontsize=24)
        pyplot.xlabel("Time", fontsize=20)
        pyplot.ylabel("Number of people", fontsize=20)
        pyplot.legend()
        return fig_total

    def dead_graph(self):
        fig_dead = pyplot.figure(figsize=(12, 12))
        pyplot.title("Dead people over time", fontsize=24)
        pyplot.plot(*zip(*self.dead_graph_list), label="Dead people")
        pyplot.xlabel("Time", fontsize=20)
        pyplot.ylabel("Number of dead people", fontsize=20)
        pyplot.legend(fontsize=16)
        return fig_dead

    def immune_graph(self):
        fig_immune = pyplot.figure(figsize=(12, 12))
        pyplot.title("Immune people over time", fontsize=24)
        pyplot.plot(*zip(*self.immune_graph_list), label="Immune people")
        pyplot.xlabel("Time", fontsize=20)
        pyplot.ylabel("Number of immune people", fontsize=20)
        pyplot.legend(fontsize=16)
        return fig_immune

    def ill_graph(self):
        fig_ill = pyplot.figure(figsize=(12, 12))
        pyplot.title("Ill people over time", fontsize=24)
        pyplot.plot(*zip(*self.ill_graph_list), label="Ill people")
        pyplot.xlabel("Time", fontsize=20)
        pyplot.ylabel("Number of ill people", fontsize=20)
        pyplot.legend(fontsize=16)
        return fig_ill

    def susceptible_graph(self):
        fig_susceptible = pyplot.figure(figsize=(12, 12))
        pyplot.title("Susceptible people over time", fontsize=24)
        pyplot.plot(*zip(*self.susceptible_graph_list), label="Susceptible people")
        pyplot.xlabel("Time", fontsize=20)
        pyplot.ylabel("Number of susceptible people", fontsize=20)
        pyplot.legend(fontsize=16)
        return fig_susceptible


    def actual_R0_graph(self):
        for element in self.average_infected():
            self.actuall_R0_list.append(element)
        fig_actual_R0 = pyplot.figure(figsize=(12, 12))
        pyplot.title("Actual R0 over time", fontsize=24)
        pyplot.plot(*zip(*self.actuall_R0_list), label="Actual R0")
        pyplot.xlabel("Time", fontsize=20)
        pyplot.ylabel("Actual R0", fontsize=20)
        pyplot.legend(fontsize=16)
        return fig_actual_R0



if __name__ == "__main__":
    analysis = Analysis()
    visualisation = Visualisation()
    susceptible_list, immune_list, ill_list, dead_list, field = analysis.create_groups()
    visualisation.animate_plot()
    if values["-death_graph-"]:
        visualisation.dead_graph()
    if values["-immune_graph-"]:
        visualisation.immune_graph()
    if values["-ill_graph-"]:
        visualisation.ill_graph()
    if values["-susceptible_graph-"]:
        visualisation.susceptible_graph()
    if values["-R0_graph-"]:
        visualisation.actual_R0_graph()
    if values["-total_cases_graph-"]:
        visualisation.total_cases_graph()
    pyplot.show()