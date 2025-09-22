import requests
from datetime import datetime, timedelta

# Imposto la data e ora desiderata (UTC)
date_str = "1979-05-28 00:00:00"

# Lista dei corpi: Sole e pianeti principali (ID HORIZONS)
# 10=Sun, 199=Mercury, 299=Venus, 399=Earth, 499=Mars, 599=Jupiter, 699=Saturn, 799=Uranus, 899=Neptune
# bodies = [10, 1, 2, 3, 4, 5, 6, 7, 8, -32]
# names = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "spacecraft"]

# Lista dei corpi: Giove e le sue lune principali + sonda
bodies = [599, 503, 504, 501, 502, 10, -32]  # Jupiter, Ganymede, Callisto, Io, Europa
names = ["Jupiter", "Ganymede", "Callisto", "Io", "Europa", "Sun", "spacecraft"]

# Funzione per ottenere dati da HORIZONS (formato vettoriale con centro il baricentro del sistema solare, J2000)
def get_horizons_vectors(body_id, date):
    url = "https://ssd.jpl.nasa.gov/api/horizons_file.api"
    
    date_dt = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    STOP_TIME = str(date_dt + timedelta(seconds=1))
    #500@0 = Solar System Barycenter
    #500@5 = Jupiter Barycenter
    # La seguente stringa considera l'insieme dei parametri richiesti da HORIZONS per ottenere i dati
    params = """!$$SOF
        MAKE_EPHEM=YES
        COMMAND=' """ + str(body_id) + """ '
        EPHEM_TYPE=VECTORS
        CENTER='500@5'
        START_TIME=' """ + str(date) + """ '
        STOP_TIME=' """ + str(STOP_TIME) + """ '
        STEP_SIZE='1 MINUTES'
        VEC_TABLE='3'
        REF_SYSTEM='ICRF'
        REF_PLANE='ECLIPTIC'
        VEC_CORR='NONE'
        CAL_TYPE='M'
        OUT_UNITS='KM-S'
        VEC_LABELS='YES'
        VEC_DELTA_T='NO'
        CSV_FORMAT='NO'
        OBJ_DATA='YES' """
    r = requests.post(url, data={'format':'text'}, files={'input': params}) # Riga di codice trovata direttamente dal sito Horizon
    if r.status_code != 200:
        raise Exception(f"Errore richiesta HORIZONS per {body_id}")
    
    # Estraggo i dati per posizioni e velocità a partire da stringhe di riferimento
    lines = r.text.splitlines()
    data_start = False
    x = y = z = vx = vy = vz = None
    for line in lines:
        if line.strip().startswith("$$SOE"):
            data_start = True
            continue
        if data_start:
            if line.strip().startswith("$$EOE"):
                break
            if "X =" in line and "Y =" in line and "Z =" in line:
                # Riga con X, Y, Z
                import re
                m = re.findall(r'([XYZ])\s*=\s*([-+]?\d+\.\d+[eE][-+]?\d+)', line)
                for label, value in m:
                    if label == 'X': x = float(value)
                    if label == 'Y': y = float(value)
                    if label == 'Z': z = float(value)
            if "VX=" in line and "VY=" in line and "VZ=" in line:
                # Riga con VX, VY, VZ
                import re
                m = re.findall(r'(V[XYZ])=\s*([-+]?\d+\.\d+[eE][-+]?\d+)', line)
                for label, value in m:
                    if label == 'VX': vx = float(value)
                    if label == 'VY': vy = float(value)
                    if label == 'VZ': vz = float(value)
                # Appena trovati tutti, restituisci
                if None not in (x, y, z, vx, vy, vz):
                    # Conversione in metri e m/s
                    return [x*1e3, y*1e3, z*1e3, vx*1e3, vy*1e3, vz*1e3]
    raise Exception(f"Dati non trovati per {body_id}")

# Estrazione dati per tutti i corpi
data = []

for i, body in enumerate(bodies):
    vec = get_horizons_vectors(body, date_str)
    data.append(vec)

# Stampa i dati raccolti in formato pronto e compatibile per la simulazione
for i, vec in enumerate(data):
    # Salva ogni valore in notazione scientifica con 15 cifre decimali
    formatted_vec = [f"{v:.15e}" for v in vec]
    if names[i] == "spacecraft":
        print(f"spacecraft[0] = [{', '.join(formatted_vec)}] # {names[i]}")
    else:
        print(f"y[{i}, 0] = [{', '.join(formatted_vec)}] # {names[i]}")

