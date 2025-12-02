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
    duration_path = os.path.join(script_dir, 'model_duration2.cbm') 
    price_path = os.path.join(script_dir, 'model_price2.cbm')     

    kmeans_model = joblib.load(kmeans_path)
    
    duration_model = CatBoostRegressor()
    duration_model.load_model(duration_path)
    
    price_model = CatBoostRegressor()
    price_model.load_model(price_path)
    
    print(f"✅ Modèles chargés : K-Means, Durée, Prix")

except Exception as e:
    print(f"⚠️ Erreur critique lors du chargement des modèles : {e}")
    kmeans_model = None
    duration_model = None
    price_model = None

# --- 1. FONCTIONS ---

def get_suggestions(text):
    """Récupère des suggestions d'adresses via l'API Photon"""
    if not text or len(text) < 3: return []
    try:
        # Bbox centrée sur NYC pour prioriser les résultats locaux
        url = f"https://photon.komoot.io/api/?q={text}&limit=5&bbox=-74.3,40.4,-73.6,40.9"
        r = requests.get(url)
        return [{'label': f"{f['properties'].get('name','')}, {f['properties'].get('city','')}", 
                 'value': f"{f['geometry']['coordinates'][1]},{f['geometry']['coordinates'][0]}--{f['properties'].get('name','')}"} 
                for f in r.json()['features']]
    except: return []

def get_route_shape(lat1, lon1, lat2, lon2):
    """Récupère le tracé réel de la route via OSRM pour l'affichage carte"""
    try:
        url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full"
        r = requests.get(url)
        data = r.json()
        if data['code'] == 'Ok':
            route = data['routes'][0]
            coords = polyline.decode(route['geometry'])
            # Retourne : lats, lons, distance_km, duree_min (trafic standard)
            return [x[0] for x in coords], [x[1] for x in coords], route['distance']/1000, route['duration']/60
    except: pass
    return [], [], 0, 0

def get_moment_category(hour):
    """Mappe l'heure aux catégories du Notebook (0=Early Morning, etc.)"""
    # Notebook logic: 0-6, 6-13, 13-16, 16-21, else Night
    if 0 <= hour < 6: return 0      # early morning
    elif 6 <= hour < 13: return 1   # morning
    elif 13 <= hour < 16: return 2  # noon/afternoon
    elif 16 <= hour < 21: return 3  # rush hour
    else: return 4                  # night

def predict_trip_data(plat, plon, dlat, dlon, date_obj):
    """
    Predit la DURÉE puis le PRIX en utilisant le chaînage des modèles.
    """
    debug_info = {"status": "Erreur - Modèles manquants", "inputs": {}, "error": None}
    
    if not duration_model or not price_model or not kmeans_model:
        return 0, 0, debug_info

    try:
        # 1. Identification des Clusters (Zones)
        p_cluster = kmeans_model.predict([[plon, plat]])[0]
        d_cluster = kmeans_model.predict([[dlon, dlat]])[0]
        
        # 2. Préparation des variables temporelles
        day_of_week = date_obj.weekday() # 0=Lundi, correspond au .cat.codes du notebook
        moment_of_day = get_moment_category(date_obj.hour)

        # 3. Construction du DataFrame pour le modèle de DURÉE
        # Le modèle a été entraîné avec : ['VendorID', 'RateCodeID', 'store_and_fwd_flag', 'payment_type', 
        #                                  'day_of_week', 'Moments_of_day', 'Pickup_Cluster', 'Dropoff_Cluster']
        # On remplit les colonnes techniques (Vendor, Rate, etc.) avec des valeurs par défaut (Mode)
        
        features_data = {
            'VendorID': [2],            # Valeur fréquente
            'RateCodeID': [1],          # Tarif standard
            'store_and_fwd_flag': [0],  # 0 pour 'N'
            'payment_type': [1],        # Carte de crédit
            'day_of_week': [int(day_of_week)],
            'Moments_of_day': [int(moment_of_day)],
            'Pickup_Cluster': [int(p_cluster)],
            'Dropoff_Cluster': [int(d_cluster)]
        }
        
        df_duration = pd.DataFrame(features_data)
        
        # --- ÉTAPE A : PRÉDICTION DURÉE ---
        pred_duration_min = duration_model.predict(df_duration)[0]
        
        # --- ÉTAPE B : PRÉDICTION PRIX ---
        # Le modèle de prix prend les MÊMES features + la durée prédite
        df_price = df_duration.copy()
        df_price['predicted_duration'] = pred_duration_min # Ajout de la colonne de liaison
        
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
        # GAUCHE : Contrôles
        html.Div([
            html.H4("1. Itinéraire", style={'margin': '0 0 10px 0'}),
            html.Label("🟩 DÉPART :", style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='pickup', placeholder="Adresse de départ...", search_value='', options=[]),
            html.Br(),
            html.Label("🟥 ARRIVÉE :", style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='dropoff', placeholder="Adresse d'arrivée...", search_value='', options=[]),
            html.Hr(),
            
            html.H4("2. Détails", style={'margin': '0 0 10px 0'}),
            
            # Slider Passagers (Uniquement pour info, l'IA V3 ne l'utilise plus, mais utile pour l'utilisateur)
            html.Label("👥 Passagers (Info) :", style={'fontWeight': 'bold', 'color':'#777'}),
            dcc.Slider(
                id='passengers-slider',
                min=1, max=6, step=1, value=1,
                marks={i: str(i) for i in range(1, 7)}
            ),
            html.Br(),

            html.Div([
                html.Div([
                    html.Label("📅 Date :"),
                    dcc.DatePickerSingle(id='date-picker', date=datetime.now().date(), display_format='DD/MM/YYYY', style={'width': '100%'})
                ], style={'flex': '1', 'marginRight': '10px'}),
                html.Div([
                    html.Label("🕒 Heure :"),
                    dcc.Dropdown(id='time-picker', options=hours_options, value=current_hour_str, clearable=False, style={'width': '100%'})
                ], style={'flex': '1'})
            ], style={'display': 'flex'}),
            
            html.Br(),
            html.Button("🚀 PRÉDIRE (Durée + Prix)", id='btn-calc', style={'width': '100%', 'padding': '15px', 'backgroundColor': '#222', 'color': 'white', 'fontWeight': 'bold', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
            
            html.Div(id='results', style={'marginTop': '20px'}),
            
            html.Details([
                html.Summary("🔧 Debug Info (Chaînage)", style={'cursor':'pointer', 'color':'#aaa'}),
                html.Pre(id='debug-output', style={'backgroundColor': '#f8f9fa', 'padding': '5px', 'fontSize': '11px'})
            ], style={'marginTop': '10px'})

        ], style={'width': '30%', 'padding': '20px', 'backgroundColor': '#fff', 'borderRadius': '10px', 'boxShadow': '0 4px 12px rgba(0,0,0,0.1)'}),

        # DROITE : Carte
        html.Div([
            dcc.Graph(id='map', style={'height': '700px', 'borderRadius': '10px'},
                      config={'scrollZoom': True},
                      figure=go.Figure(go.Scattermapbox()).update_layout(
                          mapbox_style="open-street-map", mapbox_center={"lat": 40.75, "lon": -73.98},
                          mapbox_zoom=11, margin={"r":0,"t":0,"l":0,"b":0}, showlegend=False))
        ], style={'width': '68%', 'marginLeft': '2%'})

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

@app.callback(
    [Output('map', 'figure'), Output('results', 'children'), Output('debug-output', 'children')],
    Input('btn-calc', 'n_clicks'),
    [State('pickup', 'value'), State('dropoff', 'value'),
     State('date-picker', 'date'), State('time-picker', 'value'),
     State('passengers-slider', 'value'), 
     State('map', 'figure')]
)
def calculate_trip(n, p_val, d_val, date_val, time_val, passengers, current_fig):
    if not n or not p_val or not d_val: return no_update, no_update, ""
    
    # 1. Parsing coordonnées
    plat, plon = map(float, p_val.split('--')[0].split(','))
    dlat, dlon = map(float, d_val.split('--')[0].split(','))
    
    # 2. API Route (Pour afficher la carte et comparer la distance)
    lats, lons, dist_km_osrm, dur_min_osrm = get_route_shape(plat, plon, dlat, dlon)
    
    # 3. Prédiction IA (Chaînage Duration -> Price)
    try:
         trip_datetime = datetime.strptime(f"{date_val} {time_val}", "%Y-%m-%d %H:%M")
    except:
        trip_datetime = datetime.now()

    ai_duration, ai_price, debug_data = predict_trip_data(plat, plon, dlat, dlon, trip_datetime)

    # 4. Carte
    center = current_fig['layout']['mapbox']['center']
    zoom = current_fig['layout']['mapbox']['zoom']
    fig = go.Figure()
    # Points Départ/Arrivée
    fig.add_trace(go.Scattermapbox(mode="markers", lat=[plat, dlat], lon=[plon, dlon], marker={'size': 12, 'color': ['#28a745', '#dc3545']}))
    # Tracé route
    if lats:
        fig.add_trace(go.Scattermapbox(mode="lines", lat=lats, lon=lons, line={'width': 4, 'color': '#007bff'}))
        center = {"lat": (plat + dlat)/2, "lon": (plon + dlon)/2}
    fig.update_layout(mapbox_style="open-street-map", mapbox_center=center, mapbox_zoom=zoom, margin={"r":0,"t":0,"l":0,"b":0}, showlegend=False)

    # 5. Panneau de résultats
    panel = html.Div([
        # Bloc Prix
        html.Div([
            html.H3(f"{ai_price:.2f} $", style={'color': '#28a745', 'fontSize': '42px', 'margin': '0', 'fontWeight': 'bold'}),
            html.Div("Prix Estimé (IA)", style={'color': '#666', 'fontSize': '14px', 'fontWeight':'bold'})
        ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#e8f5e9', 'borderRadius': '10px', 'marginBottom': '15px'}),
        
        # Bloc Durée Comparée
        html.Div([
            html.Div([
                html.Span("⏱️ Durée IA : ", style={'fontWeight':'bold', 'fontSize':'16px'}),
                html.Span(f"{int(ai_duration)} min", style={'color':'#d35400', 'fontWeight':'bold', 'fontSize':'18px'}),
            ], style={'marginBottom':'5px'}),
            
            html.Div([
                html.Span("🗺️ Durée GPS : ", style={'color':'#aaa', 'fontSize':'14px'}),
                html.Span(f"{int(dur_min_osrm)} min", style={'color':'#aaa', 'fontSize':'14px'}),
            ]),
        ], style={'padding':'10px', 'border':'1px solid #eee', 'borderRadius':'5px', 'marginBottom':'10px'}),

        # Bloc Infos
        html.Div([
            html.P(f"🚗 Distance : {dist_km_osrm:.2f} km", style={'color': '#555', 'margin':'2px 0'}),
            html.P(f"📅 {trip_datetime.strftime('%A %d %B à %H:%M')}", style={'color': '#888', 'fontSize':'12px', 'marginTop':'10px'})
        ])
    ], style={'padding': '20px', 'backgroundColor': 'white', 'borderLeft': '5px solid #28a745', 'boxShadow': '0 4px 15px rgba(0,0,0,0.1)', 'borderRadius': '5px'})

    return fig, panel, json.dumps(debug_data, indent=2, default=str)

if __name__ == '__main__':
    app.run(debug=True, port=8059)