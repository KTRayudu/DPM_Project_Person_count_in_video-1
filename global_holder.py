import threading
#import Live_graph_creator as initiate

lock = threading.Lock()

#Output_script, Output_div = initiate.spit_html_embedding(statistics_path='Data Files/starting_state.csv',save_locally=False)
Output_frame = None # This is a global variable which holds processed frame
Output_script =None #  This global variable holds live graph's script tag
Output_div = None # This global variable holds live graph's div tag