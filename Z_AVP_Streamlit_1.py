# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:44:31 2023

@author: Jerry
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
#from itertools import product

# Load the trained model
loaded_model = joblib.load(".APP/z_random_forest_model_202310_10_3.pkl")
#current_directory = os.path.dirname(os.path.realpath(__file__))
#model_path = os.path.join(current_directory, "z_random_forest_model_202310_10_3.pkl")
#loaded_model = joblib.load(model_path)
# Function to make predictions
def make_predictions(data):
    predictions = loaded_model.predict(data)
    return predictions

def generate_dataframe(fixed_values):
    # Define the columns
    columns = ['Toilets', 'Female Changerooms', 'Social', 'Playground', 'Outdoor Court', 'BMX',
               'Skate Park', 'Open Space/ Multi Play', 'Carpark', 'Pavillion', 'BBQ', 'Dog Park (Open)',
               'Dog Park (Fenced)', 'Flying Fox', 'Shared Path', 'Cricket Nets', 'Lighting', 'Landscaping',
               'Irrigation', 'Sports & Aquatic Centre', 'Stairs', 'Pedestrian Bridge', 'Athletics Track',
               'Field', 'Tennis Court', 'Nature Space', 'Resurfacing', 'Walking Track',
               'Sports & Recreation Reserve (all new assets)', 'Lake Stabilisation project', 'Outdoor Pool']
    
    num_rows = 100000
    
    # Create an empty DataFrame
    df_combo = pd.DataFrame(0, index=np.arange(num_rows), columns=columns)
    
    # Populate the DataFrame
    for idx in df_combo.index:
        num_columns_to_fill = np.random.randint(1, 5)
        columns_to_fill = np.random.choice(columns, num_columns_to_fill, replace=False)
        
        for col in columns_to_fill:
            df_combo.at[idx, col] = np.random.randint(1, 3)
    
    # Add Projects column
    df_combo['Projects'] = df_combo[columns].sum(axis=1)
    #df_combo.to_csv(r"E:\AXNEW\VIC LGA AVP - Copy - Copy\model building - Copy\check1.csv",index = False)
    # Fill in fixed values
    #print(fixed_values)
    for column, value in fixed_values.items():
        df_combo[column] = value
    #df_combo.to_csv(r"E:\AXNEW\VIC LGA AVP - Copy - Copy\model building - Copy\check2.csv",index = False)
    desired_columns_order = [
            'Projects', 'Toilets', 'Female Changerooms', 'Social', 'Playground', 'Outdoor Court', 'BMX',
            'Skate Park', 'Open Space/ Multi Play', 'Carpark', 'Pavillion', 'BBQ', 'Dog Park (Open)',
            'Dog Park (Fenced)', 'Flying Fox', 'Shared Path', 'Cricket Nets', 'Lighting', 'Landscaping',
            'Irrigation', 'Sports & Aquatic Centre', 'Stairs', 'Pedestrian Bridge', 'Athletics Track',
            'Field', 'Tennis Court', 'Nature Space', 'Resurfacing', 'Walking Track',
            'Sports & Recreation Reserve (all new assets)', 'Lake Stabilisation project', 'Outdoor Pool',
            'dt_TotalPopulation', 'New or Renewal_New', 'New or Renewal_New & Renewal/Upgrade',
            'New or Renewal_Renewal/Upgrade', 'Type of Open Space_Active Open Space',
            'Type of Open Space_Passive Open Space', 'Open Space Hierarchy_District',
            'Open Space Hierarchy_Local', 'Open Space Hierarchy_Regional'
        ]
    
    
    # Reorder the columns
    df_combo = df_combo[desired_columns_order]
    #print(df_combo.head())
    return df_combo



# Streamlit app
def main():
    st.title("Active Village Project Model")

    # Create input widgets for user input
    st.sidebar.header("Input Parameters")
    
    # Add input widgets for each column in your dataset
    # For example:
    #projects = int(st.sidebar.slider("Projects", 0, 10, 0,step=1))
    toilets = int(st.sidebar.slider("Toilets", 0, 5, 0,step=1))
    female_changerooms = int(st.sidebar.slider("Female Changerooms", 0, 5, 0,step=1))
    social = int(st.sidebar.slider("Social", 0, 5, 0,step=1))
    playground = int(st.sidebar.slider("Playground", 0, 5, 0,step=1))
    outdoor_court = int(st.sidebar.slider("Outdoor Court", 0, 5, 0,step=1))
    bmx = int(st.sidebar.slider("BMX", 0, 5, 0,step=1))
    skate_park = int(st.sidebar.slider("Skate Park", 0, 5, 0,step=1))
    open_space_multi_play = int(st.sidebar.slider("Open Space/ Multi Play", 0, 5, 0,step=1))
    carpark = int(st.sidebar.slider("Carpark", 0, 5, 0,step=1))
    pavillion = int(st.sidebar.slider("Pavillion", 0, 5, 0,step=1))
    bbq = int(st.sidebar.slider("BBQ", 0, 5, 0,step=1))
    dog_park_open = int(st.sidebar.slider("Dog Park (Open)", 0, 5, 0,step=1))
    dog_park_fenced = int(st.sidebar.slider("Dog Park (Fenced)", 0, 5, 0,step=1))
    flying_fox = int(st.sidebar.slider("Flying Fox", 0, 5, 0,step=1))
    shared_path = int(st.sidebar.slider("Shared Path", 0, 5, 0,step=1))
    cricket_nets = int(st.sidebar.slider("Cricket Nets", 0, 5, 0,step=1))
    lighting = int(st.sidebar.slider("Lighting", 0, 5, 0,step=1))
    landscaping = int(st.sidebar.slider("Landscaping", 0, 5, 0,step=1))
    irrigation = int(st.sidebar.slider("Irrigation", 0, 5, 0,step=1))
    sports_aquatic_centre = int(st.sidebar.slider("Sports & Aquatic Centre", 0, 5, 0,step=1))
    stairs = int(st.sidebar.slider("Stairs", 0, 5, 0,step=1))
    pedestrian_bridge = int(st.sidebar.slider("Pedestrian Bridge", 0, 5, 0,step=1))
    athletics_track = int(st.sidebar.slider("Athletics Track", 0, 5, 0,step=1))
    field = int(st.sidebar.slider("Field", 0, 5, 0,step=1))
    tennis_court = int(st.sidebar.slider("Tennis Court", 0, 5, 0,step=1))
    nature_space = int(st.sidebar.slider("Nature Space", 0, 5, 0,step=1))
    resurfacing = int(st.sidebar.slider("Resurfacing", 0, 5, 0,step=1))
    walking_track = int(st.sidebar.slider("Walking Track", 0, 5, 0,step=1))
    sports_recreation_reserve = int(st.sidebar.slider("Sports & Recreation Reserve (all new assets)", 0, 5, 0,step=1))
    lake_stabilisation_project = int(st.sidebar.slider("Lake Stabilisation project", 0, 5, 0,step=1))
    outdoor_pool = int(st.sidebar.slider("Outdoor Pool", 0, 5, 0,step=1))
    dt_total_population = int(st.sidebar.slider("dt_TotalPopulation", 0, 200000, 100000,step=1))  # Adjust the range as needed
    new_or_renewal_new = int(st.sidebar.slider("New or Renewal_New", 0, 1, 0,step=1))
    new_or_renewal_new_renewal_upgrade = int(st.sidebar.slider("New or Renewal_New & Renewal/Upgrade", 0, 1, 0,step=1))
    new_or_renewal_renewal_upgrade = int(st.sidebar.slider("New or Renewal_Renewal/Upgrade", 0, 1, 0,step=1))
    type_of_open_space_active = int(st.sidebar.slider("Type of Open Space_Active Open Space", 0, 1, 0,step=1))
    type_of_open_space_passive = int(st.sidebar.slider("Type of Open Space_Passive Open Space", 0, 1, 0,step=1))
    open_space_hierarchy_district = int(st.sidebar.slider("Open Space Hierarchy_District", 0, 1, 0,step=1))
    open_space_hierarchy_local = int(st.sidebar.slider("Open Space Hierarchy_Local", 0, 1, 0,step=1))
    open_space_hierarchy_regional = int(st.sidebar.slider("Open Space Hierarchy_Regional", 0, 1, 0,step=1))
    # Add sliders or input widgets for the remaining columns...

    # Create a DataFrame with the user's input
    projects = (
    toilets + female_changerooms + social + playground + outdoor_court +
    bmx + skate_park + open_space_multi_play + carpark + pavillion +
    bbq + dog_park_open + dog_park_fenced + flying_fox + shared_path +
    cricket_nets + lighting + landscaping + irrigation + sports_aquatic_centre +
    stairs + pedestrian_bridge + athletics_track + field + tennis_court +
    nature_space + resurfacing + walking_track + sports_recreation_reserve +
    lake_stabilisation_project + outdoor_pool
)
    # Use the calculated 'Projects' value
    st.sidebar.write(f"Projects: {projects}")
    input_data = pd.DataFrame({
        'Projects':[projects],
        'Toilets': [toilets],
        'Female Changerooms': [female_changerooms],
        'Social': [social],
        'Playground': [playground],
        'Outdoor Court': [outdoor_court],
        'BMX': [bmx],
        'Skate Park': [skate_park],
        'Open Space/ Multi Play': [open_space_multi_play],
        'Carpark': [carpark],
        'Pavillion': [pavillion],
        'BBQ': [bbq],
        'Dog Park (Open)': [dog_park_open],
        'Dog Park (Fenced)': [dog_park_fenced],
        'Flying Fox': [flying_fox],
        'Shared Path': [shared_path],
        'Cricket Nets': [cricket_nets],
        'Lighting': [lighting],
        'Landscaping': [landscaping],
        'Irrigation': [irrigation],
        'Sports & Aquatic Centre': [sports_aquatic_centre],
        'Stairs': [stairs],
        'Pedestrian Bridge': [pedestrian_bridge],
        'Athletics Track': [athletics_track],
        'Field': [field],
        'Tennis Court': [tennis_court],
        'Nature Space': [nature_space],
        'Resurfacing': [resurfacing],
        'Walking Track': [walking_track],
        'Sports & Recreation Reserve (all new assets)': [sports_recreation_reserve],
        'Lake Stabilisation project': [lake_stabilisation_project],
        'Outdoor Pool': [outdoor_pool],
        'dt_TotalPopulation': [dt_total_population],
        'New or Renewal_New': [new_or_renewal_new],
        'New or Renewal_New & Renewal/Upgrade': [new_or_renewal_new_renewal_upgrade],
        'New or Renewal_Renewal/Upgrade': [new_or_renewal_renewal_upgrade],
        'Type of Open Space_Active Open Space': [type_of_open_space_active],
        'Type of Open Space_Passive Open Space': [type_of_open_space_passive],
        'Open Space Hierarchy_District': [open_space_hierarchy_district],
        'Open Space Hierarchy_Local': [open_space_hierarchy_local],
        'Open Space Hierarchy_Regional': [open_space_hierarchy_regional],
        # Add columns for the remaining features...
    })



    fixed_values = {
        'dt_TotalPopulation': dt_total_population,
        'New or Renewal_New': new_or_renewal_new,
        'New or Renewal_New & Renewal/Upgrade': new_or_renewal_new_renewal_upgrade,
        'New or Renewal_Renewal/Upgrade': new_or_renewal_renewal_upgrade,
        'Type of Open Space_Active Open Space': type_of_open_space_active,
        'Type of Open Space_Passive Open Space': type_of_open_space_passive,
        'Open Space Hierarchy_District': open_space_hierarchy_district,
        'Open Space Hierarchy_Local': open_space_hierarchy_local,
        'Open Space Hierarchy_Regional': open_space_hierarchy_regional,
    }
    

    if st.sidebar.button("Make Predictions"):

        # Call the prediction function
        predictions = make_predictions(input_data)

        # Display the predictions
# =============================================================================
#         st.subheader("Predictions Monthly Activity Index:")
#         st.write(predictions)
# =============================================================================
        prediction_visits = round(int(predictions * 15000),0)
        predictions = round(float(predictions * 1),4)
# =============================================================================
#         with st.container():
#             st.subheader("Predictions Monthly Activity Index:")
#             st.write(predictions)
#         
# =============================================================================
        with st.container():
            st.write("---")
            left_column,right_column = st.columns(2)
            with left_column:
                st.header('Predictions Monthly Activity Index:')
                st.write('##')
                st.write(predictions)
            with right_column:
                st.header('Predictions Movement:')
                st.write('##')
                st.write(prediction_visits)            
            

        

# =============================================================================
#         st.subheader("Predictions Movement:")
#         st.write(prediction_visits)
# =============================================================================

        df_combo = generate_dataframe(fixed_values)
        
        
        # Make predictions using the loaded model
        predictions2 = make_predictions(df_combo)
        
        # Add the predictions as a new column to the 'new_data' DataFrame
        df_combo['Predictions'] = predictions2
        
        df_combo = df_combo.drop_duplicates()
        top_10_predictions = df_combo.nlargest(5, 'Predictions')
        top_10_predictions_sorted = top_10_predictions.sort_values(by='Predictions', ascending=False)
        top_10_predictions_sorted['Movement'] = round(top_10_predictions_sorted['Predictions']*15000,0)
        top_10_predictions_sorted = top_10_predictions_sorted.reset_index(drop=True)
        # Display the predictions
        st.subheader("Recommended Action:")
        def describe_row(row):
            # Extract the common features
            total_population = row['dt_TotalPopulation']
            total_projects = int(row['Projects'])
            movement = row['Movement']
        
            # List the priority features
            priority_features = [
                'New or Renewal_New', 
                'New or Renewal_New & Renewal/Upgrade',
                'New or Renewal_Renewal/Upgrade',
                'Type of Open Space_Active Open Space',
                'Type of Open Space_Passive Open Space',
                'Open Space Hierarchy_District',
                'Open Space Hierarchy_Local',
                'Open Space Hierarchy_Regional'
            ]
        
            # Extract the priority features that are not 0
            priority_values = [feature for feature in priority_features if row[feature] != 0]
        
            # Collect the other features that are not 0 (excluding the ones already extracted above and the priority features)
            other_features = [col for col in row.index 
                              if row[col] != 0 and 
                              col not in ['dt_TotalPopulation', 'Projects', 'Predictions', 'Movement'] + priority_features]
        
            # Construct the description string starting with priority values
            #description = f"Action {row.name + 1}: {', '.join(priority_values)}, dt_TotalPopulation - {total_population}. "
            
            # Add recommended actions and features
            description = f"Recommend: Total Projects: {total_projects}, "
            for feature in other_features:
                description += f"{feature}:{int(row[feature])}, "
        
            # End with Movement Change
            #description += f"Projected Movement Change: {int(movement)}"
            
            return description

        def describe_row2(row):
            # Extract the common features
            total_population = row['dt_TotalPopulation']
            total_projects = row['Projects']
            movement = row['Movement']
        
            # List the priority features
            priority_features = [
                'New or Renewal_New', 
                'New or Renewal_New & Renewal/Upgrade',
                'New or Renewal_Renewal/Upgrade',
                'Type of Open Space_Active Open Space',
                'Type of Open Space_Passive Open Space',
                'Open Space Hierarchy_District',
                'Open Space Hierarchy_Local',
                'Open Space Hierarchy_Regional'
            ]
        
            # Extract the priority features that are not 0
            priority_values = [feature for feature in priority_features if row[feature] != 0]
        
            # Collect the other features that are not 0 (excluding the ones already extracted above and the priority features)
            other_features = [col for col in row.index 
                              if row[col] != 0 and 
                              col not in ['dt_TotalPopulation', 'Projects', 'Predictions', 'Movement'] + priority_features]
        
# =============================================================================
#             # Construct the description string starting with priority values
#             description = f"Action {row.name + 1}: {', '.join(priority_values)}, dt_TotalPopulation - {total_population}. "
#             
#             # Add recommended actions and features
#             description += f"Recommend: Total Projects: {total_projects}. "
#             for feature in other_features:
#                 description += f"{feature}:{int(row[feature])}, "
#         
# =============================================================================
            # End with Movement Change
            description = f"{int(movement)}"
            
            return description

        def describe_row3(row):
            # Extract the common features
            total_population = int(row['dt_TotalPopulation'])
            total_projects = row['Projects']
            movement = row['Movement']
        
            # List the priority features
            priority_features = [
                'New or Renewal_New', 
                'New or Renewal_New & Renewal/Upgrade',
                'New or Renewal_Renewal/Upgrade',
                'Type of Open Space_Active Open Space',
                'Type of Open Space_Passive Open Space',
                'Open Space Hierarchy_District',
                'Open Space Hierarchy_Local',
                'Open Space Hierarchy_Regional'
            ]
        
            # Extract the priority features that are not 0
            priority_values = [feature for feature in priority_features if row[feature] != 0]
        
            # Collect the other features that are not 0 (excluding the ones already extracted above and the priority features)
            other_features = [col for col in row.index 
                              if row[col] != 0 and 
                              col not in ['dt_TotalPopulation', 'Projects', 'Predictions', 'Movement'] + priority_features]
        
            # Construct the description string starting with priority values
            description = f"{', '.join(priority_values)}, dt_TotalPopulation - {total_population}. "
            
# =============================================================================
#             # Add recommended actions and features
#             description += f"Recommend: Total Projects: {total_projects}. "
#             for feature in other_features:
#                 description += f"{feature}:{int(row[feature])}, "
#         
#             # End with Movement Change
#             description += f"Projected Movement Change: {int(movement)}"
# =============================================================================
            
            return description

        # Loop through each row in the dataframe and create a container for each
        for idx, row in top_10_predictions_sorted.iterrows():
            
            with st.container():
                # Display Headers and Descriptions in the two-column format
                left_column, midd_column,right_column = st.columns(3)
                
                with left_column:
                    st.write('Action:')
                    st.write('##')
                    st.write(describe_row(row))
                with midd_column:
                    st.write('Park Info:')
                    st.write('##')
                    st.write(describe_row3(row))                              
                
                with right_column:
                    st.write('Predictions Movement:')
                    st.write('##')
                    st.write(int(describe_row2(row)))
            
        # Optional: Separator between containers for better clarity
        st.write("---")                    
        
        #st.write(top_10_predictions_sorted)

        
        
if __name__ == "__main__":
    main()
