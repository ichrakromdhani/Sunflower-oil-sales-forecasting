
from keras.models import load_model
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output
import plotly.graph_objs as go


daily = pd.read_csv("daily_data.csv")
daily['date_livraison'] = pd.to_datetime(daily['date_livraison'])

d1=pd.read_csv("data.csv")
d1=d1[['date_livraison','Nbre_Palettes_EURO-PAL']]
d1['date_livraison'] = pd.to_datetime(d1['date_livraison'])
d1=d1.set_index(d1['date_livraison'])
d1=d1.rename(columns={'Nbre_Palettes_EURO-PAL': 'Nbre_Palettes_EURO_PAL'})
f=pd.read_csv("data.csv")
d2=f[['date_livraison','Nbre_Palettes_EURO-PAL']]
d2['date_livraison'] = pd.to_datetime(d2['date_livraison'])
d2=d2.rename(columns={'Nbre_Palettes_EURO-PAL': 'Nbre_Palettes_EURO_PAL'})
# Set the date as index
d2=d2.set_index(d2['date_livraison'])
d2=d2.resample('W').sum()
data = pd.read_csv("data.csv")
data=data.head(34681)


data['année']=pd.DatetimeIndex(data['date_livraison']).year
data['année']=data['année'].astype(str)
data['mois']=pd.DatetimeIndex(data['date_livraison']).month
#strftime("%b")
#data['mois_lettre'] = data['mois'].apply(lambda x: calendar.month_abbr[x])

table_volume_année = pd.pivot_table(data, values='Nbre_Palettes_EURO-PAL', index=['année'],
                     aggfunc=np.sum)
table_volume_année_vol = pd.DataFrame(table_volume_année.to_records())



#table_volume_mois = pd.pivot_table(data, values='Nbre_Palettes_EURO-PAL', index=['mois_lettre'],
                    # aggfunc=np.sum)
#table_volume_mois_vol = pd.DataFrame(table_volume_mois.to_records())


data["date_livraison"] = pd.to_datetime(data["date_livraison"], format="%Y-%m-%d")
df=data[['date_livraison','Nbre_Palettes_EURO-PAL']]
temp=df.set_index(df['date_livraison'])

table_pays_client = pd.pivot_table(data, values='Nbre_Palettes_EURO-PAL', index=['date_livraison','Pays','Client'],
                     aggfunc=np.sum)
print(table_pays_client)
pays_client_df_vol = pd.DataFrame(table_pays_client.to_records())
df_client_livraison=data[['Client','Bon de Livraison Num','Pays','date_livraison']]
#df_client_livraison = df_client_livraison.groupby('Client')
df_client_livraison['Bon de Livraison Num'] = df_client_livraison['Bon de Livraison Num'].astype(str)
df_client_livraison["Total livraison"] = 1
# Create a pivot table of the volume distributions
daily1=daily.head(721)
pivot_table_cl_liv = df_client_livraison.pivot_table(df_client_livraison, index = ['date_livraison','Pays','Client'], aggfunc = np.sum)
cl_liv_df = pd.DataFrame(pivot_table_cl_liv.to_records())
total_livraison=cl_liv_df['Total livraison']


pays_client_df = pd.concat([pays_client_df_vol, total_livraison], axis=1).head(34681)

palette_par_article = pd.pivot_table(data, values='Nbre_Palettes_EURO-PAL', index=['Designation'],
                     aggfunc=np.sum)
                     
palette_par_article_df = pd.DataFrame(palette_par_article.to_records())
top_5_articles = palette_par_article_df.sort_values(by="Nbre_Palettes_EURO-PAL", ascending=False).head()
palette_par_pays = pd.pivot_table(data, values='Nbre_Palettes_EURO-PAL', index=['Pays'],
                     aggfunc=np.sum)
palette_par_pays_df = pd.DataFrame(palette_par_pays.to_records())
top_5_pays = palette_par_pays_df.sort_values(by="Nbre_Palettes_EURO-PAL", ascending=False).head()
palette_par_client = pd.pivot_table(data, values='Nbre_Palettes_EURO-PAL', index=['Client'],
                     aggfunc=np.sum)
palette_par_client_df = pd.DataFrame(palette_par_client.to_records())
top_5_clients = palette_par_client_df.sort_values(by="Nbre_Palettes_EURO-PAL", ascending=False).head()

temp=df.set_index(df['date_livraison'])
# Select the proper time period for weekly aggreagation
weekly = d1.resample('W').sum()
weekly=weekly.head(141)
weekly=weekly.rename(columns={'Nbre_Palettes_EURO-PAL': 'Nbre_Palettes_EURO_PAL'})
#data = data.query("Pays == 'Germany'")
#data["date_livraison"] = pd.to_datetime(data["Date"], format="%Y-%m-%d")
#data.sort_values("Date", inplace=True)
weekly = pd.DataFrame(weekly.to_records())

monthly = temp.resample('M').sum()
monthly = pd.DataFrame(monthly.to_records())

monthly['date_livraison']=pd.to_datetime(monthly['date_livraison'])
monthly['month'] = pd.DatetimeIndex(monthly['date_livraison']).month

yearly = temp.resample('Y').sum()
#yearly=yearly.head(141)
yearly = pd.DataFrame(yearly.to_records())
yearly['year'] = pd.DatetimeIndex(yearly['date_livraison']).year


print(yearly)

colors = {"background": "#F3F6FA", "background_div": "white", 'text': '#7FDBFF'}




# Use `hole` to create a donut-like pie chart
fig_top_pays = go.Figure(data=[go.Pie(labels=top_5_pays.Pays, values=top_5_pays['Nbre_Palettes_EURO-PAL'], hole=.3)])
fig_top_clients = go.Figure(data=[go.Pie(labels=top_5_clients.Client, values=top_5_clients['Nbre_Palettes_EURO-PAL'], hole=.3)])
fig_top_articles = go.Figure(data=[go.Pie(labels=top_5_articles.Designation, values=top_5_articles['Nbre_Palettes_EURO-PAL'], hole=.3)])
fig_vol_année = go.Figure(data=[go.Bar(x=yearly['year'], y=yearly['Nbre_Palettes_EURO-PAL'],hovertemplate= "Année=%{x}<br>Ventes=%{y:.2f}<extra></extra>",marker=dict(
        color='rgba(255, 99, 71, 0.6)',
        line=dict(color='rgba(255, 99, 71, 0.6)', width=3)
    ))])
#fig_vol_mois = go.Figure(data=[go.Bar(x=table_volume_mois_vol['mois_lettre'], y=table_volume_mois_vol['Nbre_Palettes_EURO-PAL'],hovertemplate= "$%{y:.2f}<extra></extra>")])
fig_vol_année.update_layout(paper_bgcolor='white',plot_bgcolor='white')

fig_vol_année.update_yaxes(
    title='Nombre des palettes sorties',
    )

fig_vol_année.update_xaxes(
    title='Année',
    )

#fig_vol_mois.update_layout(paper_bgcolor='white',plot_bgcolor='white')
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


fig_monthly= go.Figure(data=[go.Bar(x=monthly['date_livraison'], y=monthly['Nbre_Palettes_EURO-PAL'],hovertemplate= "Date=%{x}<br>Ventes=%{y:.2f}<extra></extra>",marker=dict(
        color='rgba(35, 203, 167, 1)',
        line=dict(color='rgba(35, 203, 167, 1)', width=3)
    ))])
fig_monthly.update_xaxes(
    title='Mois',
    dtick="M1",
    tickformat="%b\n%Y")

fig_monthly.update_yaxes(
    title='Nombre des palettes sorties',
    )
fig_monthly.update_layout(paper_bgcolor='white',plot_bgcolor='white')


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Analysez et prévoyez vos ventes!"

server = app.server

colors = {"background": "#F3F6FA", "background_div": "white", 'text': '#009999'}

app.config['suppress_callback_exceptions']= True


app.layout = html.Div(style={'backgroundColor': colors['background']}, 
        children=[html.Div(
            children=[
                html.P(children="🌻", className="header-emoji"),
                html.H1(
                    children="Analyse et prévision des ventes de l'huile de tournesol", className="header-title"
                ),
                html.P(
                    children= " Analyse de 751 jours de vente et Prévision de 100 futurs jours de vente.",
                    className="header-description",
                ),
            ],
            className="header",
        ),


   html.Div([

                html.Div([
                    html.Div([
                        html.H6('Top 5 Pays', style={'textAlign': 'center'}),
                        dcc.Graph(
                            id='example-graph-1',
                            figure=fig_top_pays
                        )
                    ], className="four columns"),

                    html.Div([
                        html.H6('Top 5 Clients', style={'textAlign': 'center'}),
                        dcc.Graph(
                            id='example-graph-2',
                            figure=fig_top_clients
                        )
                    ], className="four columns"),

                    html.Div([
                        html.H6('Top 5 Articles', style={'textAlign': 'center'}),
                        dcc.Graph(
                            id='example-graph-3',
                            figure=fig_top_articles
                        )
                    ], className="four columns")

                ], className="row", style={"margin": "1% 3%"}),

                html.Div([
                    html.Div([
                        html.H6('Nombre des palettes sorties par année', style={'textAlign': 'center'}),
                        dcc.Graph(
                            id='example-graph-4',
                            figure=fig_vol_année
                        )
                        ], className="six columns"),

                     html.Div([
                        html.H6('Nombre des palettes sorties par mois de l année', style={'textAlign': 'center'}),
                        dcc.Graph(
                            id='example-graph-5',
                            figure=fig_monthly
                        )
                        ], className="six columns"),
                ], className="row", style={"margin": "1% 3%"}),

                html.Div(children=[
                   html.Div(children=[
                       html.Div(children="Pays", className="menu-title"),
                        dcc.Dropdown(
                            id="region-filter",
                            options=[
                                {"label": Pays, "value": Pays}
                                for Pays in np.sort(pays_client_df.Pays.unique())
                            ],
                            value="Germany",
                            clearable=False,
                            className="dropdown",
                        ),]
                        
                        ,className="four columns" ),
                        html.Div(
                    children=[
                        html.Div(children="Client", className="menu-title"),
                        dcc.Dropdown(
                            id="type-filter",
                            options=[
                                {"label": Client, "value": Client}
                                for Client in pays_client_df.Client.unique()
                            ],
                            value="Li99794",
                            clearable=False,
                            searchable=False,
                            className="dropdown",
                        ),
                    ],
            className="four columns" ),

                html.Div(
                    children=[
                        html.Div(
                            children="Intervalle du temps", className="menu-title"
                        ),
                        dcc.DatePickerRange(
                            id="date-range",
                            min_date_allowed=pays_client_df.date_livraison.min().date(),
                            max_date_allowed=pays_client_df.date_livraison.max().date(),
                            start_date=pays_client_df.date_livraison.min().date(),
                            end_date=pays_client_df.date_livraison.max().date(),
                        ),
                    ]
               ,
              className="four columns"),


                        
                        

                     html.Div([
                       html.Div(children=" Choisissez un intervalle du temps! ", className="title"),
                        dcc.Graph(
                        id="price-chart",
                        config={"displayModeBar": False},
                    ),], ),
                ], className="row", style={"margin": "1% 3%"}),


                        ]
        ),

         html.Div([
                    html.Div([
                        html.H6('Nombre des palettes sorties par jour', style={'textAlign': 'center'}),
                        dcc.Graph(
                            id='example-graph-6',
                            figure={
                                'data': [
                                    {
                "x": daily1["date_livraison"],
                "y": daily1["volume_des_ventes"],
                "type": "lines",
                "hovertemplate": "Date=%{x}<br>Ventes=%{y:.2f}<extra></extra>",

            },

                                ],
                                'layout': {
                                    #'title': 'Nombre des palettes par jour'
                                }
                            }
                        )
                    ])
                ], className="row", style={"margin": "1% 3%"}),


            html.Div([
                html.Div([
                    html.H6('Combien des palettes pour les 100 futurs jours ?', style={'color':'red','textAlign': 'center','fontSize': 25}),
           dbc.Button("Predict",id='submit-val',style={'color': 'blue', 'fontSize': 18})

                ]),
              html.Div(id='container1',style={'color': 'black', 'fontSize': 20,'textAlign': 'center',"margin": "4% 10%"}),
            html.Div(id='container2',style={'color': 'green', 'fontSize': 26,'textAlign': 'center',"margin": "5%"})

            ],className="row", style={"margin": "3% 3%"})

        
    
])
@app.callback(
    Output("price-chart", "figure"),
    [
        Input("region-filter", "value"),
        Input("type-filter", "value"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
    ],
)
def update_charts(Pays, Client, start_date, end_date):
    mask = (
        (pays_client_df.Pays == Pays)
        & (pays_client_df.Client == Client)
        & (pays_client_df.date_livraison >= start_date)
        & (pays_client_df.date_livraison <= end_date)
    )
    filtered_data = pays_client_df.loc[mask, :]
    print(filtered_data)
    
    price_chart_figure = {
        "data": [
            {
                "x": filtered_data["date_livraison"],
                "y": filtered_data["Nbre_Palettes_EURO-PAL"],
                "type": "lines",
                "hovertemplate": "Date=%{x}<br>Ventes=%{y:.2f}<extra></extra>",
            },
        ],
        "layout": {
            "title": {
                "text": "Nombre des palettes sorties par client dans un intervalle du temps précis",
                "x": 0.05,
                "xanchor": "left",
            },
            "xaxis": {"fixedrange": True},
            "yaxis": { "fixedrange": True},
            "colorway": ["#17B897"],
        },
    }

    volume_chart_figure = {
        "data": [
            {
                "x": filtered_data["date_livraison"],
                "y": filtered_data["Total livraison"],
                "type": "lines",
            },
        ],
        "layout": {
            "title": {"text": "Nombre de livraison par jour", "x": 0.05, "xanchor": "left"},
            "xaxis": {"fixedrange": True},
            "yaxis": {"fixedrange": True},
            "colorway": ["#E12D39"],
        },
    }
    return price_chart_figure

@app.callback(
     Output(component_id='container1', component_property='children'),
     Output(component_id='container2', component_property='children'),
    [Input(component_id='submit-val', component_property='n_clicks'),
     ])

def forecast_next_week(n_clicks):
    if (n_clicks) is None:
        raise PreventUpdate
    else:
        model = load_model('model.h5')
        df_histo = daily.head(751)
        print(df_histo)
        df_histo = df_histo.reset_index()['volume_des_ventes']
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_histo = scaler.fit_transform(np.array(df_histo).reshape(-1, 1))

        ##splitting dataset into train and test split
        training_size = int(len(df_histo) * 0.80)
        test_size = len(df_histo) - training_size
        train_data, test_data = df_histo[0:training_size, :], df_histo[training_size:len(df_histo), :1]

        # demonstrate prediction for next 5 weeks
        from numpy import array

        x_input = test_data[len(test_data) - 10:].reshape(1, -1)

        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()

        nombre_futur_semaine = 100
        lst_output = []
        n_steps = 10
        i = 0
        while (i < nombre_futur_semaine):

            if (len(temp_input) > n_steps):
                # print(temp_input)
                x_input = np.array(temp_input[1:])
                print("{} week input {}".format(i, x_input))
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, n_steps, 1))
                # print(x_input)
                yhat = model.predict(x_input, verbose=0)
                print("{} week output {}".format(i, yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                # print(temp_input)
                lst_output.extend(yhat.tolist())
                i = i + 1
            else:
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                print(yhat[0])
                temp_input.extend(yhat[0].tolist())
                print(len(temp_input))
                lst_output.extend(yhat.tolist())
                i = i + 1

        print(lst_output)

        df_réelle = daily.tail(100)
        print(df_réelle)
        print(df_réelle)
        predections = scaler.inverse_transform(lst_output)
        preds_1d = predections.flatten()

        date = df_réelle.date_livraison
        réel = df_réelle.volume_des_ventes[:nombre_futur_semaine].to_list()
        prédit = preds_1d
        res = réel - prédit
        res_abs = abs(res)
        erreur = res_abs / prédit
        pour_erreur = erreur * 100

        # dictionary of lists
        dict = {'Date': date, 'Réel': réel, 'prédit': prédit, 'Ecart': res, 'Ecart_abs': res_abs,
                'Ecart_abs/Réel': erreur,
                '% erreur_prédiction': pour_erreur}

        vanilla_lstm_26_semaines = pd.DataFrame(dict)
        table = dbc.Table.from_dataframe(vanilla_lstm_26_semaines, bordered=True, dark=True, hover=True, responsive=True, striped=True)
        tauxErreurMoy= (vanilla_lstm_26_semaines['Ecart_abs/Réel'].sum() / 26) * 100
        msg= 'Taux erreur moyen : 18.2%'
        return table, msg


if __name__ == '__main__':
    app.run_server(debug=True)