import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pingouin as pg
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.formula.api import ols



@st.cache_data
def get_summary_data():
    timepoint_cda = pd.read_csv('data/avgCDA_sep_timepoints_persubj.csv', index_col=0)
    avg_cda = pd.read_csv('data/avgCDApersubj.csv', index_col=0)
    data_behav = pd.read_csv('data/subjK_setsize.csv', index_col=0)
    p3_time = pd.read_csv('data/Pelec_0to500.csv')
    p3_block = pd.read_csv('data/Pelec_250to500_block_cor_setsize.csv')

    return timepoint_cda, avg_cda, data_behav, p3_time, p3_block


header = st.container()
dataset = st.container()
results = st.container()

with header:
    st.title('The Role of CDA and P300 in Visual Working Memory: Insights from a Change Detection Task')
    # st.divider()
    st.header('Abstract', divider='blue')
    st.write(
        '''Working memory (WM) is a complex cognitive system responsible for temporarily retaining information for active manipulation. The study primarily focuses on understanding the neural mechanisms underlying visual working memory capacity using insights generated from the electroencephalograph (EEG) recorded data. We look at two well-studied event-related potentials: (i) contralateral delay activity (CDA) and (ii) P300 components, while replicating the procedure of the Vogel and Machizawa, 2004 study.''')
    st.write(
        'Utilizing a lateralized change detection task with three distinct conditions (2, 4, and 6 set sizes), the study analyzed EEG recordings while the focus was on discerning patterns in CDA amplitude across different set sizes. Additionally, we also evaluate the attention of the participants during the encoding phase of the trials by studying the P300 component for each condition of the task. We looked to unravel the effect of attention in terms of fatigue by observing the amplitude and latency changes over the duration of the experiment.')
    st.write(
        "The findings revealed intriguing trends in CDA amplitude, reflecting the complexity of visual working memory tasks. The P300 component's analysis added a new dimension to existing theories, challenging conventional associations with memory load and response inhibition. These insights contribute to a broader understanding of cognitive processing and visual working memory, shedding light on the multifaceted nature of these neural components.")
    st.divider()

with dataset:
    timepoint_cda, avg_cda, data_behav, p3_time, p3_block = get_summary_data()
    # st.write(timepoint_cda.describe())
    st.header("Data obtained from participants:", divider='blue')
    df = timepoint_cda.copy()
    df.columns = ['Subject', 'Set Size', 'Time point', 'CDA']
    num_rows_to_display = st.slider("Number of Rows to Display", min_value=5, max_value=(len(df)), step=5)
    # User input for selecting a specific 'Set Size'
    set_sizes = df['Set Size'].unique().astype(str)  # Convert to strings
    participants = df['Subject'].unique().astype(str)
    # User input for selecting a specific 'Set Size'
    default_option = "All"
    selected_set_size = st.selectbox("Select a Set Size:", [default_option] + list(set_sizes))
    selected_participant = st.selectbox('Select a Participant:', [default_option] + list(participants))
    if (selected_set_size == default_option) & (selected_participant == default_option):
        filtered_data = df
    elif (selected_set_size == default_option) & (selected_participant != default_option):
        filtered_data = df[df['Subject'] == int(selected_participant)]
    elif (selected_participant == default_option) & (selected_set_size != default_option):
        filtered_data = df[df['Set Size'] == int(selected_set_size)]
    else:
        filtered_data = df[(df['Set Size'] == int(selected_set_size)) & (df['Subject'] == int(selected_participant))]


    # Display the filtered DataFrame
    st.table(filtered_data[:num_rows_to_display])
    # Calculating descriptive statistics for each set size
    descriptive_stats = timepoint_cda.groupby('setsize').agg({
        'meanCDA': ['mean', 'std', 'min', 'max', 'count']}).reset_index()

    # Renaming columns for better readability
    descriptive_stats.columns = ['Set Size', 'Mean CDA', 'Standard Deviation', 'Min CDA', 'Max CDA', 'Count']

    # Displaying the descriptive statistics
    st.subheader('Observing the distribution of CDA activity for each set size:')
    st.table(descriptive_stats.assign(hack='').set_index('hack'))
    st.write(
        'We used custom MATLAB scripts along with some EEGLAB functions and ZapLine to pre-process the data. The table shows how the pre-processed data looks for each paricipant with respect to the EEG activity obtained from a 300-900ms window averaged across all trials. ')
    st.divider()

with results:
    st.header('1. Behavioural Analyses: ', divider='blue')
    subjects = data_behav['subj'].unique().astype(str)
    selected_subj = st.multiselect('Choose a Subject to see their Average Memory Capacity (K): ', default=subjects[::3], options=([default_option] + list(subjects)))
    if default_option in selected_subj:
        subj_filtered = data_behav
    else:
        selected_subj = [int(i) for i in selected_subj if i.isdigit()]
        subj_filtered = data_behav[data_behav['subj'].isin(selected_subj)]

    st.table(subj_filtered)
    st.markdown('''The behavioral data consists of two columns:
- `subj`: The subject identifier.
- `meank`: The memory capacity \( K \) for each subject, calculated as $ K = S*(H - F) $, where \(S\) is the size of the array, \(H\) is the observed hit rate, and \(F\) is the false alarm rate.

#### Analysis Steps:

1. **Filter Participants**: Reject participants with \( K < 1 \).
2. **Summary Statistics**: Compute summary statistics for the memory capacity \( K \) to understand its distribution.
3. **Visualize the Data**: Plot the distribution of \( K \) to visually inspect the data.
4. **Relate with CDA Activity**: If needed, we can explore the relationship between \( K \) and CDA amplitude to understand the connection between memory capacity and neural activity.''')
    # Filter participants with K < 1
    behavioral_data = data_behav[data_behav['meank'] >= 1]

    # Create a Plotly histogram
    fig_behav = go.Figure(data=[go.Histogram(x=behavioral_data['meank'], nbinsx=10)])

    # Customize the layout
    fig_behav.update_layout(
        title='Distribution of Memory Capacity (K) for Participants with K >= 1',
        xaxis_title='Memory Capacity (K)',
        yaxis_title='Frequency',
        title_x=0.2,
        title_y=0.95
    )
    st.plotly_chart(fig_behav)
    st.markdown(
        '''**Figure 1.** The distribution of memory capacity (𝐾) for participants with (𝐾≥1) is depicted in the histogram.''')
    st.write(
        'The memory capacity (K) of participants was calculated using the formula mentioned earlier using the behavioural data recorded from the key presses on each trial and their respective array sizes. Post calculation of the K scores, the participants that had a value of K<1 were discarded as they were assumed to be highly distracted during the experiment thus producing such a low behavioural performance score. The distribution of the K scores is shown in Fig.1 for the 11 participants after discarding 5 participants due to their K scores being below 1. The mean memory capacity (K) of the sample is 1.71 and the K ranges from 1.01 to 2.50.')
    st.divider()

    st.subheader('Relationship between CDA activity and K-scores')
    st.write(
        'In addition to computing the memory capacity (K), we also explored the relationship between the change in K scores and CDA amplitudes for each of the participants on each set size as shown in Fig. 2 using Pearson’s correlation (r). However, the analysis did not provide strong evidence (see Table 1 for r and p values) for a significant relationship between CDA activity and memory capacity (K) in the context of this change detection task (p>0.05). ')
    # Merge the CDA data with the filtered behavioral data containing K values
    merged_data = pd.merge(avg_cda, behavioral_data[['subj', 'meank']], on='subj', how='inner')
    # Create a figure for the scatter plots
    fig_behav_corr = px.scatter(merged_data, x='meanCDA', y='meank', color='setsize',
                                title='CDA vs Memory Capacity (K)')

    # Store correlation results
    correlation_results = {}

    # Iterate through set sizes and calculate correlation and visualize the relationship
    for setsize in merged_data['setsize'].unique():
        subset_data = merged_data[merged_data['setsize'] == setsize]
        correlation_coefficient, p_value = stats.pearsonr(subset_data['meanCDA'], subset_data['meank'])
        correlation_results[setsize] = (correlation_coefficient, p_value)

        # Add scatter plot trace to the figure
        fig_behav_corr.add_trace(
            go.Scatter(x=subset_data['meanCDA'], y=subset_data['meank'], mode='markers', name=f'Set Size {setsize}',
                       text=f'Set Size {setsize}: r = {correlation_coefficient:.2f}'))

    # Customize the layout
    fig_behav_corr.update_xaxes(title='Mean CDA Amplitude')
    fig_behav_corr.update_yaxes(title='Memory Capacity (K)')
    fig_behav_corr.update_layout(showlegend=True, coloraxis_showscale=False, title_x=0.35, title_y=0.95)

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig_behav_corr)
    st.markdown(
        '''**Figure 2.** These scatter plots depict the relationship between the mean CDA amplitude of each participant and memory capacity (K) for each set size (2, 4, 6). ''')
    correlation_data = {
        'Set Size': list(correlation_results.keys()),
        'r-score': [result[0] for result in correlation_results.values()],
        'p-value': [result[1] for result in correlation_results.values()]}
    df_corr = pd.DataFrame(correlation_data)
    st.table(df_corr.assign(hack='').set_index('hack'))
    st.markdown(
        '''**Table 1.** The table shows the results for Pearson’s correlation (r) and p-values of CDA amplitude with K-scores for each set size.''')

    st.header('2. Analyses of Event-Related Potentials (ERPs): ', divider='blue')
    st.subheader('CDA Analyses: ')
    st.write(
        'The CDA analysis was computed for the time window between 300-900ms. The initial distribution displayed a varying CDA amplitude for each of the set sizes as seen in the Fig. 3a.')

    # Set Seaborn context and palette
    plt.style.use('default')
    sns.set_context("paper")
    custom_palette = sns.color_palette("magma", len(timepoint_cda['setsize'].unique()))
    sns.set_palette(custom_palette)

    # Create a figure and axis
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    fig1.set_facecolor((14 / 255, 17 / 255, 23 / 255, 1))
    ax1.set_facecolor((14 / 255, 17 / 255, 23 / 255, 1))

    # Plotting the mean CDA for each set size
    for set_size in timepoint_cda['setsize'].unique():
        subset_data = timepoint_cda[timepoint_cda['setsize'] == set_size]
        sns.lineplot(data=subset_data, x='timepoint', y='meanCDA', label=f'Set Size {set_size}')

    # Customize the labels
    ax1.spines['bottom'].set_color('gray')
    ax1.spines['top'].set_color('gray')
    ax1.spines['left'].set_color('gray')
    ax1.spines['right'].set_color('gray')
    ax1.set_title('a) Averaged Contralateral Delay Activity (CDA) for Each Set Size',
                  fontdict={'fontsize': 16, 'fontweight': 'bold'}, color=(250 / 255, 250 / 255, 250 / 255, 1))
    ax1.set_xlabel('Time Point (ms)', fontsize=12, color='white')
    ax1.set_ylabel('Mean CDA Amplitude (\u03BCV)', fontsize=12, color='white')
    plt.tick_params(axis='both', colors='white')
    plt.legend(labelcolor='white', facecolor=(14 / 255, 17 / 255, 23 / 255, 1))

    st.pyplot(fig1)

    # Plotly boxplot for CDA amplitude range for each set size
    fig2 = px.box(avg_cda, x='setsize', y='meanCDA', color='setsize',
                  labels={'meanCDA': 'Mean CDA Amplitude'},
                  title='b) Distribution of CDA across each Set Size',
                  width=700, height=500)

    # Customize the layout
    fig2.update_layout(
        xaxis_title='Set Size',
        yaxis_title='Mean CDA Amplitude (\u03BCV)',
        showlegend=False,  # Hide the legend
        title_x=0.30,
        title_y=0.95,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    st.plotly_chart(fig2)

    st.write(
        '**Figure 3. a)** The averaged CDA activity observed during the retention phase of the trials for each set size between 300ms and 900ms after the onset of the memory array, **b)** The distribution of mean CDA amplitude across all participants for each set size.')
    st.write('''The mean CDA amplitudes were found to be -0.401, -0.414, and -0.576 for the set sizes 2, 4, and 6 respectively. The distribution in Fig. 3b shows the variability in the mean values of the CDA across all participants. We implemented a one-way repeated measures ANOVA to investigate whether there is a significant difference in contralateral delay activity (CDA) across different set sizes (2, 4, 6) in the change detection task. The repeated measures were chosen as the same participants are subjected to different set sizes and the dependence between measurements needs to be accounted for, i.e., the variability within subjects.
''')
    st.markdown('''
    #### Analysis Steps:
    
    1. **Assumptions**: Since the same subjects are used for each set size (a within-subject design), and we want to test if there are differences in mean CDA amplitude across these levels, repeated measures ANOVA is appropriate.
    2.  **Normality and sphericity**: Been checked and met.
    3. **Hypothesis Testing:**
       * **Null Hypothesis \(H_0\)**: There is no significant difference in the mean CDA across the different set sizes (2, 4, 6 set sizes).
       * **Alternative Hypothesis \(H_1\)**: There is a significant difference in the mean CDA across the different set sizes.
    ''')

    # Calculate the mean CDA for each set size
    set_size_plot = avg_cda.groupby('setsize')['meanCDA'].mean().reset_index()

    # Create a line graph with markers using Plotly Express
    fig_ss_cda = px.line(set_size_plot, x='setsize', y='meanCDA', markers=True, line_shape='linear',
                         labels={'setsize': 'Memory Array Size', 'meanCDA': 'Mean Amplitude'},
                         title='Mean CDA Amplitude vs Memory Array Size')

    # Customize the layout
    fig_ss_cda.update_xaxes(title_text='Memory Array Size', tickvals=[2, 4, 6])
    fig_ss_cda.update_yaxes(title_text='Mean Amplitude (\u03BCV)', autorange='reversed')
    fig_ss_cda.update_layout(title_x=0.3, title_y=0.95)

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig_ss_cda)
    st.markdown(
        '''**Figure 4.** The graph shows the average distribution of mean CDA amplitude averaged over each set size indicating an increase in CDA over each successive set size.''')
    # Perform repeated measures ANOVA
    rm_anova_result = pg.rm_anova(timepoint_cda, dv='meanCDA', within='setsize', subject='subj')
    pg_anova = pd.DataFrame(rm_anova_result)

    # Print the result
    st.table(pg_anova.assign(hack='').set_index('hack'))
    st.markdown(
        '''**Table 2.** The results of the one-way repeated measures ANOVA. (ddof1 = degrees of freedom for the numerator (between-groups), ddof2 = degrees of freedom for the denominator (within-groups), F = F-value, p = p-value, ng2 = generalized eta-squared effect size, eps = Greenhouse-Geisser epsilon factor).''')
    st.write('''The F-statistic gives a value of 1.307, representing the ratio of the between-group variance to the within-group variance, indicating a large difference between the means. However, the p-value (0.293) was not significant (p>0.05), indicating that we fail to reject the null hypothesis that the CDA is the same across the different set sizes.

The results suggest that there is no statistically significant difference in CDA amplitude across the set sizes (2, 4, 6) in this sample. The small effect size further supports the conclusion that set size does not have a substantial impact on CDA amplitude in the context of the experiment conducted.
''')
    st.divider()
    st.subheader('P300 Analyses: ')
    # Define the set sizes
    set_sizes = [2, 4, 6]

    # Create subplots for each condition
    fig_eeg = go.Figure()

    for set_size in set_sizes:
        # Filter data for the specific set size
        subset_data = p3_time[p3_time['setsize'] == set_size]

        # Group by timepoint and calculate the mean amplitude
        mean_amplitude = subset_data.groupby('timepoint')['mval'].mean()

        # Add a trace for the mean amplitude
        fig_eeg.add_trace(go.Scatter(x=mean_amplitude.index, y=mean_amplitude.values, mode='lines',
                                     name=f'Set Size {set_size}'))

    # Customize the layout
    fig_eeg.update_layout(
        title='a) EEG Signal Activity for Different Set Sizes',
        xaxis_title='Timepoint (ms)',
        yaxis_title='Mean Amplitude (\u03BCV)',
        title_x=0.3,
        title_y=0.95
    )
    # Invert the y-axis
    fig_eeg.update_yaxes(autorange="reversed")
    # Remove grid lines
    fig_eeg.update_xaxes(showgrid=False)
    fig_eeg.update_yaxes(showgrid=False)

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig_eeg)

    # Group the data by set size and calculate the mean P300 amplitude for each condition
    p300_condition = p3_block.groupby('setsize')['mval'].mean().reset_index()

    # Create a bar chart using Plotly Express
    fig_p3box = px.bar(p300_condition, x='setsize', y='mval',
                       labels={'setsize': 'Set Size', 'mval': 'Mean Amplitude (\u03BCV)'},
                       title='b) P300 Amplitude by Set Size')

    # Customize the layout
    fig_p3box.update_xaxes(title_text='Set Size')
    fig_p3box.update_yaxes(title_text='Mean Amplitude (\u03BCV)')
    fig_p3box.update_layout(title_x=0.37, title_y=0.95)

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig_p3box)
    st.markdown(''' **Figure 5. a)** EEG signal from 0 to 500ms after the stimulus onset. **b)** P300 mean amplitude for each set size.

''')
    st.write(
        'The P300 component was analyzed by taking the mean amplitude in the time window of 250-500 ms after the stimulus onset for each set size (Fig 5b).  The initial analysis showed varying mean P300 amplitudes of 0.953, 0.496, and 0.524 for each set size respectively. The P300 component analysis of the change detection task to observe the impact of attention on the working memory capacity was composed of three objectives: ')
    st.subheader('Examination of the P300 amplitude for correct vs incorrect trials:')
    # Create a Plotly figure for each set size
    # Map the correctness values to labels
    p3_block['correctness_label'] = p3_block['cor'].map({0: 'Incorrect', 1: 'Correct'})

    # Calculate the mean P300 amplitude for each set size and correctness, averaged across all blocks and participants
    mean_p300_data = p3_block.groupby(['setsize', 'correctness_label'])['mval'].mean().reset_index()

    # Define custom colors for correct and incorrect bars
    colors = {'Incorrect': '#FF6F63', 'Correct': '#4BDCFF'}

    # Create a bar chart with Plotly Express
    fig_p300 = px.bar(mean_p300_data, x='setsize', y='mval', color='correctness_label',
                      title='P300 Amplitude for Correct and Incorrect Trials for Each Condition',
                      labels={'mval': 'P300 Amplitude'},
                      color_discrete_map=colors,
                      category_orders={'correctness_label': ['Incorrect', 'Correct']})

    # Customize the layout
    fig_p300.update_xaxes(title_text='Set Size', tickvals=[2,4,6])
    fig_p300.update_yaxes(title_text='P300 Amplitude (\u03BCV)')
    fig_p300.update_layout(legend_title_text='Trial Response', title_x=0.17, title_y=0.95 )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig_p300)
    st.markdown('''**Figure 6.** Bar plot showing the P300 amplitude for correct and incorrect trials on each set size.''')
    st.write('The initial descriptive analysis of the mean P300 amplitude is shown in Fig 5b. The P300 amplitude is higher on correct than incorrect trials for set sizes 4 and 6, but lower on set size 2.')
    st.write('A 2-way repeated measures ANOVA was performed to examine whether the P300 amplitude is higher for correct than incorrect trials across each set size. Again, the repeated measures were chosen to account for the within-subject variability as the same subjects were exposed to all three conditions. We also looked for any interaction effects between the correctness (correct vs incorrect) and set size (2,4, and 6).')
    st.markdown('''     
     ##### Analysis Plan:
     **Dependent Variable**: P300 Amplitude (mean amplitude within the 250 to 500 ms time window)
     **Independent Variable**: Correctness (correct vs. incorrect trials)
     **Grouping Variable**: Set Size (2, 4, 6)
     **Statistical Test**: Two-Way Repeated Measures ANOVA with Set Size and Correctness as within-subject factors
     **Within-Subject Factors**: Correctness (correct or incorrect) and Set Size (set sizes 2, 4, and 6)
     **Main Effects**: Correctness, Set Size
     **Interaction Effect**: Correctness × Set Size
                ''')
    # Perform a two-way repeated measures ANOVA with correctness and set size as within-subject factors
    two_way_anova_result = pg.rm_anova(dv='mval', within=['cor', 'setsize'], subject='subj', data=p3_block)
    df_p3corrinc = pd.DataFrame(two_way_anova_result)
    st.table(df_p3corrinc.assign(hack='').set_index('hack'))
    st.markdown('Table 3. The table shows the results for the two-way repeated measures ANOVA with Set Size and Correctness (Cor) as within-subject factors (_ddof1 = degrees of freedom for the numerator (between-groups), ddof2 = degrees of freedom for the denominator (within-groups), F = F-value, p = p-value, ng2 = generalized eta-squared effect size, eps = Greenhouse-Geisser epsilon factor_).')
    st.write('The correctness effect result (F(1,13) = 0.0822, p = 0.7789) shows there is no significant main effect of correctness on the P300 amplitude. This means that the data does not support the hypothesis that the P300 amplitude is significantly higher for correct trials compared to incorrect trials. The set size effect shows (F(2,26) = 0.2177, p = 0.8058) there is no significant main effect of set size on the P300 amplitude. This indicates that the data does not show a significant difference in P300 amplitude across the different set sizes. The interaction effects show (F(2,26) = 0.7808, p = 0.4685) there is no significant interaction between correctness and set size. This means that the effect of correctness on P300 amplitude does not significantly vary across different set sizes. ')
    st.write('The results do not provide evidence to support the hypothesis that the P300 amplitude is higher for correct than incorrect trials for each set size condition. Both the main effects (correctness and set size) and the interaction effect (correctness x set size) were found to be non-significant.')
    st.write('These findings may be indicative of the complexity of the underlying cognitive processes and the factors that influence P300 amplitude. Additional analyses or a more refined experimental design may be required to further investigate the relationships between correctness, set size, and P300 amplitude.')
    st.divider()
    st.subheader('Examination of the P300 amplitude for each set size as a function of the block number')
    fig_p3, ax_p3 = plt.subplots(figsize=(12, 6))
    fig_p3.set_facecolor((14 / 255, 17 / 255, 23 / 255, 1))
    ax_p3.set_facecolor((14 / 255, 17 / 255, 23 / 255, 1))
    sns.lineplot(data=p3_block, x='block', y='mval', hue='setsize', marker='o')
    plt.xticks([1, 2, 3, 4, 5], rotation=0)
    plt.grid(False)

    # Customize the labels
    ax_p3.spines['bottom'].set_color('gray')
    ax_p3.spines['top'].set_color('gray')
    ax_p3.spines['left'].set_color('gray')
    ax_p3.spines['right'].set_color('gray')
    ax_p3.set_title('P300 Amplitude Across Blocks for Each Set Size',
                    fontdict={'fontsize': 16, 'fontweight': 'bold'}, color=(250 / 255, 250 / 255, 250 / 255, 1))
    ax_p3.set_xlabel('Block Number', fontsize=12, color='white')
    ax_p3.set_ylabel('P300 Amplitude Amplitude (\u03BCV)', fontsize=12, color='white')
    plt.tick_params(axis='both', colors='white')
    plt.legend(labelcolor='w', facecolor=(14 / 255, 17 / 255, 23 / 255, 1))

    st.pyplot(fig_p3)
    st.markdown('**Figure 7.** The distribution of the P300 amplitude over each successive block for each set size.')
    st.write('To examine if the duration of the experiment had any impact on the processing capacity of the participants, we looked at the change in P300 amplitude across each block on each successive set size (Fig 7).')
    st.write('We also examined whether the P300 amplitude shows a gradual decline across each set size as a function of block number, again using a 2-way repeated measures ANOVA, with the blocks (1-5) and set sizes (2, 4, and 6) as within-group factors.')
    st.markdown('''
    #### Analysis Steps:
    1. Dependent Variable: P300 Amplitude
        * The mean amplitude of the P300 component within the time window of 250 to 500 ms.
        * This variable represents the response or outcome we are analyzing.
    2. Within-Subject Factors:
        * Set-size (condition): The number of colored sqaures show on the trial.
        * Levels: 2, 4, and 6
    3. Block (block): Block number in the experiment.
    4. Levels: 1, 2, 3, 4, and 5
    ''')
    # Calculate the mean P300 amplitude for each participant, set size, and block
    anova_hyp2 = p3_block.groupby(['subj', 'setsize', 'block'])['mval'].mean().reset_index()
    # Perform a two-way repeated measures ANOVA with correctness and set size as within-subject factors
    two_way_anova_result_2 = pg.rm_anova(dv='mval', within=['setsize', 'block'], subject='subj', data=anova_hyp2)
    st.table(two_way_anova_result_2.assign(hack='').set_index('hack'))
    st.markdown('''
    **Table 4.** The table shows the results for the two-way repeated measures ANOVA with Set Size and Blocks as within-subject factors. (_ddof1 = degrees of freedom for the numerator (between-groups), ddof2 = degrees of freedom for the denominator (within-groups), F = F-value, p = p-value, ng2 = generalized eta-squared effect size, eps = Greenhouse-Geisser epsilon factor_).
    ''')
    st.write('The analysis did not find a statistically significant difference in P300 amplitude across any of the set sizes (F(2, 28) = 0.8557, p = 0.4358). This indicates that the mean amplitude of the P300 component did not vary significantly based on the set size condition. Additionally, The analysis did not find a statistically significant change in P300 amplitude across blocks 1 to 5 (F(4, 56) = 0.4987, p = 0.7367). This indicates that the mean amplitude of the P300 component did not vary significantly across the five blocks. The interaction between the set size and the blocks is not significant. This result suggests that the relationship between set size and P300 amplitude does not vary significantly across blocks (F(8, 112) = 0.9225, p = 0.5009).')
    st.write('The 2-way repeated measures ANOVA did not reveal significant effects of set size or block number on the P300 amplitude within the time window of 250 to 500ms. The findings suggest that neither the set size condition nor the progression of blocks significantly influenced the P300 amplitude in this dataset. Specifically, the analysis did not support the hypothesis that the P300 amplitude shows a gradual decline across each set size as a function of block number. ')
    st.divider()
    st.subheader('Examine the P300 latency as a function of the block number for each set size')
    st.write('We also looked at the decline in processing capacity through the P300 latency (Fig 8). If there is a decrease in attention, P300 latency should increase as a function of the block number for each of the set sizes.')

    # Filter the data to include only the time window of 250 to 500 ms
    p300_latency = p3_time[(p3_time['timepoint'] >= 250) & (p3_time['timepoint'] <= 500)]

    # Group the data by set size, block, and time point, and calculate the mean amplitude for each group
    p300_latency_mean = p300_latency.groupby(['setsize', 'block', 'timepoint'])['mval'].mean().reset_index()

    # Identify the time point at which the P300 amplitude is at its maximum for each set size and block (P300 latency)
    p300_latency_peak = p300_latency_mean.loc[p300_latency_mean.groupby(['setsize', 'block'])['mval'].idxmax()]

    # Rename the columns for clarity
    p300_latency_peak.rename(columns={'timepoint': 'P300_latency', 'mval': 'P300_peak_amplitude'}, inplace=True)

    fig_p3lat, ax_p3lat = plt.subplots(figsize=(12, 6))
    fig_p3lat.set_facecolor((14 / 255, 17 / 255, 23 / 255, 1))
    ax_p3lat.set_facecolor((14 / 255, 17 / 255, 23 / 255, 1))
    sns.lineplot(data=p300_latency_peak, x='block', y='P300_latency', hue='setsize', marker='o')
    plt.xticks([1, 2, 3, 4, 5], rotation=0)
    plt.grid(False)
    ax_p3lat.set_title('Mean P300 Latency Across Blocks for Each Set Size', fontdict={'fontsize': 16, 'fontweight': 'bold'}, color=(250 / 255, 250 / 255, 250 / 255, 1))
    ax_p3lat.spines['bottom'].set_color('white')
    ax_p3lat.spines['left'].set_color('white')
    ax_p3lat.spines['right'].set_color('white')
    ax_p3lat.spines['top'].set_color('white')
    ax_p3lat.set_xlabel('Block Number', fontsize=12, color='white')
    ax_p3lat.set_ylabel('Mean P300 Latency (ms)', fontsize=12, color='white')
    plt.tick_params(axis='both', colors='white')
    plt.legend(labelcolor='w', facecolor=(14 / 255, 17 / 255, 23 / 255, 1))
    st.pyplot(fig_p3lat)
    st.markdown('**Figure 8.** The plot shows the distribution of the mean P300 latency on all blocks for each set size.')
    st.write('Linear regression was used to model the relationship between block number (independent variable) and P300 latency (dependent variable) for each set size. The slope of this relationship provided a direct test of whether P300 latency increases with the block number.')
    st.markdown('''
    ##### Analysis Steps:
    1. **Data Preparation**: Organize the data into separate subsets for each set size, with block number and P300 latency as variables.
    2. **Linear Regression Analysis**: Perform a linear regression analysis for each set size to assess the relationship between block number and P300 latency.
    3. **Interpret Results**: Examine the slope and significance of the regression to determine whether there is evidence that P300 latency increases with block number for each set size.
    ''')
    # Create an empty list to store DataFrames
    results_list = []

    # Perform linear regression for each set size
    for set_size in set_sizes:
        # Subset the data for the specific set size
        subset_data = p300_latency_peak[p300_latency_peak['setsize'] == set_size]

        # Perform linear regression with block as the independent variable and P300_latency as the dependent variable
        regression_model = ols('P300_latency ~ block', data=subset_data).fit()

        # Get the slope, intercept, and p-value for the block coefficient
        slope = regression_model.params['block']
        intercept = regression_model.params['Intercept']
        p_value = regression_model.pvalues['block']

        # Create a DataFrame for the current set size
        result_df = pd.DataFrame(
            {'Set Size': [set_size], 'Slope': [slope], 'Intercept': [intercept], 'p-value': [p_value]})

        # Append the DataFrame to the list
        results_list.append(result_df)

    # Concatenate all DataFrames into one
    regression_results = pd.concat(results_list, ignore_index=True)
    st.table(regression_results.assign(hack='').set_index('hack'))
    st.markdown('''**Table 6.** The table shows the results of the linear regression.''')
    st.write('The results obtained from the linear regression analysis (Table 6) do not reveal any significant evidence to support the hypothesis that P300 latency increases as a function of block number for any of the set sizes. For set sizes 2 and 6, the slopes were -2.4 and -19.6 which is non-significant and the p-value>0.05 for all three set sizes. These findings suggest that, from the given data that was analyzed, there is no systematic increase in P300 latency across the block number for any of the set sizes over the course of the experiment. ')
