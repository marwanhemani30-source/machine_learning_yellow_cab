import dash
from dash import dcc, html, Input, Output, State, no_update
import requests
import plotly.graph_objects as go
import polyline
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor
from datetime import datetime
import os
import json

app = dash.Dash(__name__)

# --- 0. CHARGEMENT DES MODÈLES ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kmeans_path = os.path.join(script_dir, 'kmeans_clusters.pkl')
    duration_path = os.path.join(script_dir, 'model_duration.cbm') 
    price_path = os.path.join(script_dir, 'model_price.cbm')     
    kmeans_model = joblib.load(kmeans_path)

    duration_model = CatBoostRegressor()
    duration_model.load_model(duration_path)
    price_model = CatBoostRegressor()
    price_model.load_model(price_path)

    # price_model = joblib.load(os.path.join(script_dir, 'model_price_lightgbm.pkl')) for LBGM
    
    print(f"Models loaded successfully.")

except Exception as e:
    print(f"Loading Failure : {e}")
    kmeans_model = None
    duration_model = None
    price_model = None

# --- 1. FONCTIONS ---

def aget_suggestions(text):
    """Real-time address suggestions from Photon API"""
    if not text or len(text) < 3: return []
    try:
        # Focus on NYC area
        url = f"https://photon.komoot.io/api/?q={text}&limit=5&bbox=-74.3,40.4,-73.6,40.9"
        r = requests.get(url)
        return [{'label': f"{f['properties'].get('name','')}, {f['properties'].get('city','')}", 
                 'value': f"{f['geometry']['coordinates'][1]},{f['geometry']['coordinates'][0]}--{f['properties'].get('name','')}"} 
                for f in r.json()['features']]
    except: return []

def get_suggestions(text):
    if not text or len(text) < 3: return []
    
    
    url = f"https://photon.komoot.io/api/?q={text}&limit=5&bbox=-74.3,40.4,-73.6,40.9"
    
    # 2. Mask as a browser 
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        print(f"Tentative de recherche pour : {text}") # Debug
        
        
        r = requests.get(url, headers=headers, timeout=5)
        
        print(f"Code retour API : {r.status_code}") # Debug: 403 or 429: blocked!
        
        if r.status_code == 200:
            return [{'label': f"{f['properties'].get('name','')}, {f['properties'].get('city','')}", 
                     'value': f"{f['geometry']['coordinates'][1]},{f['geometry']['coordinates'][0]}--{f['properties'].get('name','')}"} 
                    for f in r.json()['features']]
        else:
            return []
            
    except Exception as e:
        print(f"Error : {e}")
        return []

def get_route_shape(lat1, lon1, lat2, lon2):
    """Dynamic path from OSRM API"""
    try:
        url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full"
        r = requests.get(url)
        data = r.json()
        if data['code'] == 'Ok':
            route = data['routes'][0]
            coords = polyline.decode(route['geometry'])
            
            return [x[0] for x in coords], [x[1] for x in coords], route['distance']/1000, route['duration']/60
    except: pass
    return [], [], 0, 0

def get_moment_category(hour):
    """Time of day categorization for model input"""
    if 0 <= hour < 6: return 0      # early morning
    elif 6 <= hour < 13: return 1   # morning
    elif 13 <= hour < 16: return 2  # noon/afternoon
    elif 16 <= hour < 21: return 3  # rush hour
    else: return 4                  # night

def predict_trip_data(plat, plon, dlat, dlon, date_obj):
    """
    Time preidiction and Price prediction using chained models.
    """
    debug_info = {"status": "Erreur - Missing models", "inputs": {}, "error": None}
    
    if not duration_model or not price_model or not kmeans_model:
        return 0, 0, debug_info

    try:
       
        p_cluster = kmeans_model.predict([[plon, plat]])[0] # Pickup cluster
        d_cluster = kmeans_model.predict([[dlon, dlat]])[0] # Dropoff cluster
        
        day_of_week = date_obj.weekday() 
        moment_of_day = get_moment_category(date_obj.hour)


        features_data = {
            'payment_type': [1],          
            'day_of_week': [int(day_of_week)],
            'Moments_of_day': [int(moment_of_day)],
            'Pickup_Cluster': [int(p_cluster)],
            'Dropoff_Cluster': [int(d_cluster)]
        }
        
        df_duration = pd.DataFrame(features_data) # Artificial DataFrame prediction step
        
        # Time prediction using pretrained model
        pred_duration_min = duration_model.predict(df_duration)[0]
        
        # Price prediction using time prediction
        df_price = df_duration.copy()
        df_price['predicted_duration'] = pred_duration_min 
        
        pred_price_usd = price_model.predict(df_price)[0]

        debug_info = {
            "status": "Succès",
            "cluster_depart": int(p_cluster),
            "cluster_arrivee": int(d_cluster),
            "moment": moment_of_day,
            "duree_predite": pred_duration_min,
            "prix_predit": pred_price_usd
        }

        return pred_duration_min, pred_price_usd, debug_info

    except Exception as e:
        debug_info["status"] = "Erreur Python"
        debug_info["error"] = str(e)
        return 0, 0, debug_info

# --- 2. INTERFACE ---

hours_options = [{'label': f"{h:02d}:00", 'value': f"{h:02d}:00"} for h in range(24)]
current_hour_str = f"{datetime.now().hour:02d}:00"

app.layout = html.Div([
    html.H2("🚖 NYC Taxi - AI Predictor V2", style={'textAlign': 'center', 'fontFamily': 'sans-serif', 'color': '#333'}),
    
    html.Div([
        
        html.Div([
            html.H4("1. Path", style={'margin': '0 0 10px 0'}),
            html.Label("START :", style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='pickup', placeholder="Starting address...", search_value='', options=[]),
            html.Br(),
            html.Label("ARRIVAL :", style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='dropoff', placeholder="Arrival address...", search_value='', options=[]),
            html.Hr(),
            
            html.H4("2. Details", style={'margin': '0 0 10px 0'}),
            
            
            html.Label("Passengers Number :", style={'fontWeight': 'bold', 'color':'#777'}),
            dcc.Slider(
                id='passengers-slider',
                min=1, max=6, step=1, value=1,
                marks={i: str(i) for i in range(1, 7)}
            ),
            html.Br(),

            html.Div([
                html.Div([
                    html.Label("Date :"),
                    dcc.DatePickerSingle(id='date-picker', date=datetime.now().date(), display_format='DD/MM/YYYY', style={'width': '100%'})
                ], style={'flex': '1', 'marginRight': '10px'}),
                html.Div([
                    html.Label("Pickup Time :"),
                    dcc.Dropdown(id='time-picker', options=hours_options, value=current_hour_str, clearable=False, style={'width': '100%'})
                ], style={'flex': '1'})
            ], style={'display': 'flex'}),
            
            html.Br(),
            html.Button("Find Results HERE", id='btn-calc', style={'width': '100%', 'padding': '15px', 'backgroundColor': '#222', 'color': 'white', 'fontWeight': 'bold', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
            
            html.Div(id='results', style={'marginTop': '20px'}),
            
            html.Details([
                html.Summary("Debug Info", style={'cursor':'pointer', 'color':'#aaa'}),
                html.Pre(id='debug-output', style={'backgroundColor': '#f8f9fa', 'padding': '5px', 'fontSize': '11px'})
            ], style={'marginTop': '10px'})

        ], style={'width': '30%', 'padding': '20px', 'backgroundColor': '#fff', 'borderRadius': '10px', 'boxShadow': '0 4px 12px rgba(0,0,0,0.1)'}),

        
        html.Div([
            dcc.Loading(
                id="loading-map",
                type="default",  # Options: 'graph', 'cube', 'circle', 'dot', 'default'
                color="#28a745", # On reprend le vert de votre thème
                children=[
                    dcc.Graph(id='map', style={'height': '700px', 'borderRadius': '10px'},
                              config={'scrollZoom': True},
                              figure=go.Figure(go.Scattermapbox()).update_layout(
                                  mapbox_style="open-street-map", mapbox_center={"lat": 40.75, "lon": -73.98},
                                  mapbox_zoom=11, margin={"r":0,"t":0,"l":0,"b":0}, showlegend=False))
                ]
            )
        ], style={'width': '68%', 'marginLeft': '2%', 'position': 'relative'})

    ], style={'display': 'flex', 'maxWidth': '1400px', 'margin': 'auto', 'padding': '10px'})
])

# --- 3. CALLBACKS ---

def ensure_value_in_options(search_results, current_value):
    if not current_value: return search_results
    if any(opt['value'] == current_value for opt in search_results): return search_results
    try:
        label = current_value.split('--')[1]
        return [{'label': label, 'value': current_value}] + search_results
    except: return search_results

@app.callback(Output('pickup', 'options'), Input('pickup', 'search_value'), State('pickup', 'value'))
def update_pickup_options(search, value):
    results = get_suggestions(search) if search else []
    return ensure_value_in_options(results, value)

@app.callback(Output('dropoff', 'options'), Input('dropoff', 'search_value'), State('dropoff', 'value'))
def update_dropoff_options(search, value):
    results = get_suggestions(search) if search else []
    return ensure_value_in_options(results, value)

# --- CALLBACK 1 : RAPIDE (IA + PRIX + TEMPS) ---
@app.callback(
    [Output('results', 'children'), Output('debug-output', 'children')],
    Input('btn-calc', 'n_clicks'),
    [State('pickup', 'value'), State('dropoff', 'value'),
     State('date-picker', 'date'), State('time-picker', 'value')]
)
def update_prediction_panel(n, p_val, d_val, date_val, time_val):
    if not n or not p_val or not d_val: return no_update, ""
    
    # Parsing
    plat, plon = map(float, p_val.split('--')[0].split(','))
    dlat, dlon = map(float, d_val.split('--')[0].split(','))
    
    # Date
    try: trip_datetime = datetime.strptime(f"{date_val} {time_val}", "%Y-%m-%d %H:%M")
    except: trip_datetime = datetime.now()

    # APPEL IA (Rapide)
    ai_duration, ai_price, debug_data = predict_trip_data(plat, plon, dlat, dlon, trip_datetime)
    
    # Construction du Panneau (Sans Distance textuelle)
    panel = html.Div([
        # PRIX
        html.Div([
            html.H3(f"{ai_price:.2f} $", style={'color': '#28a745', 'fontSize': '42px', 'margin': '0', 'fontWeight': 'bold'}),
            html.Div("Estimated Price", style={'color': '#666', 'fontSize': '14px', 'fontWeight':'bold'})
        ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#e8f5e9', 'borderRadius': '10px', 'marginBottom': '15px'}),
        
        # DURÉE
        html.Div([
            html.Div([
                html.Span("Predicted Time : ", style={'fontWeight':'bold', 'fontSize':'16px'}),
                html.Span(f"{int(ai_duration)} min", style={'color':'#d35400', 'fontWeight':'bold', 'fontSize':'18px'}),
            ], style={'marginBottom':'5px'}),
        ], style={'padding':'10px', 'border':'1px solid #eee', 'borderRadius':'5px', 'marginBottom':'10px'}),

        # INFO DATE
        html.P(f"Date: {trip_datetime.strftime('%d/%m/%Y à %H:%M')}", style={'color': '#888', 'fontSize':'12px'})
    ], style={'padding': '20px', 'backgroundColor': 'white', 'borderLeft': '5px solid #28a745', 'boxShadow': '0 4px 15px rgba(0,0,0,0.1)', 'borderRadius': '5px'})

    return panel, json.dumps(debug_data, indent=2, default=str)


# --- CALLBACK 2 : LENT (CARTE + ROUTE API) ---
@app.callback(
    Output('map', 'figure'),
    Input('btn-calc', 'n_clicks'),
    [State('pickup', 'value'), State('dropoff', 'value'),
     State('map', 'figure')]
)
def update_map_only(n, p_val, d_val, current_fig):
    if not n or not p_val or not d_val: return no_update
    
    # Parsing
    plat, plon = map(float, p_val.split('--')[0].split(','))
    dlat, dlon = map(float, d_val.split('--')[0].split(','))
    
    # Appel API OSRM (Prend du temps, mais ne bloque pas le prix)
    lats, lons, _, _ = get_route_shape(plat, plon, dlat, dlon)
    
    # Mise à jour Carte
    center = current_fig['layout']['mapbox']['center']
    zoom = current_fig['layout']['mapbox']['zoom']
    fig = go.Figure()

    # Points Départ/Arrivée
    fig.add_trace(go.Scattermapbox(
        mode="markers", lat=[plat, dlat], lon=[plon, dlon], 
        marker={'size': 12, 'color': ['#28a745', '#dc3545']}
    ))
    
    # Tracé Bleu
    if lats:
        fig.add_trace(go.Scattermapbox(
            mode="lines", lat=lats, lon=lons, 
            line={'width': 4, 'color': '#007bff'}
        ))
        center = {"lat": (plat + dlat)/2, "lon": (plon + dlon)/2}
        
    fig.update_layout(
        mapbox_style="open-street-map", mapbox_center=center, mapbox_zoom=zoom, 
        margin={"r":0,"t":0,"l":0,"b":0}, showlegend=False
    )
    
    return fig

if __name__ == '__main__':
    app.run(debug=True, port=8059)