MARCAS_VALIDAS = [
    'ford', 'jeep', 'volkswagen', 'chevrolet', 'renault', 'toyota',
    'peugeot', 'nissan', 'citroen', 'bmw', 'honda', 'hyundai', 'audi',
    'fiat', 'chery', 'kia', 'mercedesbenz', 'dodge', 'baic', 'suzuki',
    'porsche', 'landrover', 'mitsubishi', 'volvo', 'ds', 'ssangyong',
    'alfaromeo', 'jac', 'jetour', 'gwm', 'isuzu', 'lifan',
    'lexus', 'subaru', 'daihatsu', 'mini', 'kaiyi', 'jaguar'
]


MODELOS_POR_MARCA = {
    'ford': ['ecosport', 'territory', 'kuga', 'broncosport', 'explorer', 'bronco'],
    'chevrolet': ['tracker', 'trailblazer', 'equinox', 'spin', 'blazer', 'grandblazer', 'grandvitara'],
    'peugeot': ['2008', '3008', '4008'], 'renault': ['duster', 'captur', 'dusteroroch', 'koleos', 'sanderostepway'],
    'jeep': ['compass', 'renegade', 'grandcherokee', 'commander', 'wrangler', 'cherokee', 'patriot'],
    'nissan': ['kicks', 'xtrail', 'murano', 'pathfinder', 'xterra', 'terranoii'], 
    'volkswagen': ['taos', 'tcross', 'tiguan', 'tiguanallspace', 'touareg', 'nivus'], 
    'kaiyi': ['x3'], 'toyota': ['corollacross', 'hiluxsw4', 'sw4', 'rav4', 'landcruiser', '4runner'], 
    'citroen': ['c4cactus', 'c3aircross', 'c5aircross', 'c3', 'c4aircross'], 
    'hyundai': ['tucson', 'santafe', 'creta', 'x35', 'galloper', 'kona', 'grandsantafe'], 
    'fiat': ['pulse', '500x'], 'honda': ['hrv', 'crv', 'pilot'], 
    'bmw': ['x1', 'x3', 'x5', 'x6', 'x4', 'x2', 'serie4'], 
    'audi': ['q5', 'q3', 'q2', 'q7', 'q3sportback', 'q8', 'sq5', 'q5sportback'], 
    'kia': ['sportage', 'soul', 'sorento', 'seltos', 'mohave'], 
    'baic': ['x55', 'x25', 'x35'], 'jac': ['s2'], 
    'mercedesbenz': ['claseglc', 'clasegla', 'clasegle', 'claseglk', 'claseml', 'clasegl', 'ml'], 
    'chery': ['tiggo', 'tiggo3', 'tiggo4pro', 'tiggo5', 'tiggo2', 'tiggo4', 'tiggo8pro'], 
    'dodge': ['journey'], 'landrover': ['evoque', 'rangeroversport', 'discovery', 'rangerover', 'freelander', 'defender'], 
    'suzuki': ['grandvitara', 'vitara', 'jimny', 'samurai'], 'porsche': ['cayenne', 'macan', 'panamera'], 
    'volvo': ['xc60', 'xc40'], 
    'ds': ['ds7crossback', 'ds7', 'ds3'], 
    'ssangyong': ['musso', 'actyon'], 'alfaromeo': ['stelvio'], 'jetour': ['x70'], 
    'gwm': ['jolion', 'h6'], 'isuzu': ['trooper'], 'lifan': ['myway', 'x70'], 
    'lexus': ['ux', 'nx'], 'subaru': ['outback'], 'daihatsu': ['terios'], 
    'mini': ['coopercountryman'], 'mitsubishi': ['outlander', 'montero', 'nativa'], 'jaguar': ['fpace']}




TERMINOS_TRACCION = {
    '4x4': ['4x4','awd','4matic','quattro','4m','4wd','xdrive'],
    '4x2': ['2x4','4x2','fwd','rwd','2wd']
}

TERMINOS_TRANSMISION = {
    'automatica': [
        'at', '6at', '8at', 'at6', 'atx', 'cvt', 'tiptronic', 'stronic',
        'dsg', 'automatica', 'automatic', 'tronic', 'aut', 'automatico', 'automático'
    ],
    'manual': ['mt', '6mt', 'manual']
}

TIPOS_COMBUSTIBLE = {
    'Eléctrico': ['electrico', 'electrica', 'electric', 'electrical'],
    'Híbrido': ['hibrido', 'hibrid', 'hybrid', 'hibrida', 'mhev', 'hev', 'phev', 'hv', 'mild hybrid'],
    'Diésel': ['diesel', 'gasoil'],
    'Nafta': ['nafta', 'naftero'],
    'Nafta/GNC': ['gnc']
}
