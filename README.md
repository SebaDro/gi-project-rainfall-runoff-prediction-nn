# GI-Projekt KI SoSe 2021 - Rainfall Runoff Prediction mit Neuronalen Netzen
In diesem zweiten kleineren Projekt soll ein Neuronales Netz zur Vorhersage des Niederschlagsabfluss in einem Einzugsgebiet
unter Nutzung der Deep Learning Bibliothek Tensorflow trainiert werden. Hierzu sollen insbesondere Techniken und konzeptionelle
Verfahrensweisen des Machine Learnings eigenständig angewendet werden.

## Vorbereitung
### Einrichtung der Data Science Umgebung
Nutzen Sie die bereitgestellte `environment.yml` um eine separate Environment in Conda zu erstellen. Die `environment.yml`
enthält neben der Deep Learning Bibliothek [TensorFlow](https://www.tensorflow.org/) weitere Dependencies für statistische
Bibliotheken.
### Daten
Wie in den Übungen zuvor, erfolgt die Bearbeitung der Aufgaben auf Grundlage meteorologischer und hydrologischer
Zeitreihen für Einzugsgebiete aus dem CAMELS-US Datensatz. Die Daten für einige ausgewählte Einzugsgebiete liegen im
_./data_ Ordner vor. Falls Sie die Modellierung auf selbst gewählten Einzugsgebieten durchführen möchten, finden Sie
den gesamten CAMELS-US Datensatz unter: https://ral.ucar.edu/solutions/products/camels. Hier reicht es aus den ersten
Datensatz (_CAMELS time series meteorology, observed flow, meta data (.zip)_) herunterzuladen.

## Vorgaben
Kapseln Sie eigene Implementierungen soweit wie möglich sinnvoll in separaten Modulen, Klassen und Methoden (denkbar wären z.B. eine
mit sinnvollen Variablen und Methoden ausgestattete Klasse `CamelsDataset` zur Kapselung eines CAMEL-US Datensatzes oder eine Klasse
 `RrpModel` mit Methoden zum Trainieren und Auswerten eines Neuronalen Netzes mit Tensorflow). Hiermit stellen Sie die Wiederverwendbarkeit
 in nachfolgenden Übungen und Projekten sicher. Die Anwendung der eigenen Klassen und Module für die o.g. Aufgabe soll anschließend
 innerhalb eines Jupyter Notebooks erfolgen. Beschreiben Sie dabei die einzelnen Schritte und gehen Sie insbesondere bei der Auswertung
 auf die Modellierungsergebnisse mit Hilfe entsprechender Visualisierungen ein.

## Aufgabe
Mit Tensorflow soll ein vollvernetztes Neuronales Netz für die Vorhersage des Niederschlagsabflusses auf Grundlage
meteorologischer und hydrologischer Daten modelliert werden. Ziel ist es, dass das trainierte Modelle in der Lage ist,
aus den Niederschlagswerten der letzten n-Tage den Abfluss des nächsten Tages vorherzusagen. Dabei sollen Training,
Validierung und Testen des Modells jeweils auf einem anderen Zeitabschnitt der vorliegenden Messdatenreihe eines
ausgewählten Einzugsgebiets durchgeführt werden. Kalibrieren Sie das trainierte Modell unter Nutzung des
Validierungsdatensatzes und Evaluieren Sie das Modell abschließend mit dem Testdatensatz. Gehen Sie dabei wie folgt vor:

### 1. Trainings-, Validierungs- und Testdatensätze auswählen  
Erstellen Sie für ein ausgewähltes Einzugsgebiet aus den gesamten vorhandenen Zeitreihendatensätzen drei sich nicht 
überschneidende Teilmengen, die Sie als Trainings-, Validierungs- und Testdatensätze verwenden. Dabei sollten die
Validierungs- und Testdaten jeweils einen Zeitabschnitt abdecken, der hinter dem Zeitabschnitt der Trainingsdaten liegt.
Folgende Verhältnisse der drei Teilmengen zueinander sind sinnvoll: 80/10/10, 70/15/15, 70/10/20. 

### 2. Feature Engineering  
**Normierung**  
Skalieren Sie die Niederschlags- und Abflusswerte, so dass diese einheitlich Werte zwischen 0 und 1 annehmen. Berücksichtigen 
Sie, dass die Abflusswerte zunächst ins metrische System transformiert werden müssen. Außerdem benötigen Sie die
Skalierungsfaktoren der Abflusswerte später noch für eine Rücktransformation der vorhergesagten Werte. 

**Input- und Output-Tensoren**  
Erstellen Sie für die Input Features sowie für die Zielwerte (Targets) entsprechende NumPy Tensoren. Als Features sollen
die Niederschlagswerte der letzten n-Tage (einschließlich des aktuellen Tages) verwendet werden. Als Zielwert wird jeweils
der Abfluss des nächsten Tages (n+1) verwendet. Würde man dies tabellarisch darstellen, ergibt sich:  

| sample     | prec (t=1)| prec (t=2)  | prec (t=3)    | prec (t=4)    | ... |prec (t=n)     |discharge (t=n+1)|
|------------|-----------|-------------|---------------|---------------|-----|---------------|-----------------|
| 0          | 0.68      | 0.75        | 0.45          | 0.87          | ... | 0.56          | 0.54            |
| 1          | 0.75      | 0.45        | 0.87          | 0.92          | ... | 0.71          | 0.62            |
| 2          | 0.45      | 0.87        | 0.92          | 0.38          | ... | 0.76          | 0.78            |
| ...        | ...       | ...         | ...           | ...           | ... | ...           | ...             |

Die Tensoren der Input-Features und der Zielwerte sollten somit folgendes aussehen besitzen:
Input Shape: (samples, features)
Target Shape: (samples, 1)

Beispiel: Für einen Datensatz über ein gesamtes Jahr (365 Tage) sollen die Niederschlagswerte der letzten 50 Tage  
genutzt werden, um den Niederschlag des nächsten Tages vorherzusagen. Der Input Tensor besitzt dann die Shape (315,50)
und der Tensor der Targets die Shape (315,1).

### 3. Modell Training
Trainieren Sie mit den zuvor erstellten Tensoren für Input Features und Targets ein Model in Tensorflow/Keras.
Erstellen Sie hierfür ein vollvernetztes Neuronales Netz mit mehreren Hidden Layern. Ein solches Netz wird in Tensorflow bzw. Keras
mit einem Sequentiellen Modell (https://www.tensorflow.org/api_docs/python/tf/keras/Sequential) erstellt, das mehrere 
Dense Layer (https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense) besitzt. 

Treffen Sie bezüglich des Aufbaus des Neuronalen Netzes für folgende Konfigurationsparameter eine sinnvolle Auswahl:
* Anzahl der Hidden Layer
* Anzahl der Neuronen in den einzelnen Layern
* Aktivierungsfunktion (https://www.tensorflow.org/api_docs/python/tf/keras/activations)
* Loss-Funktion (https://www.tensorflow.org/api_docs/python/tf/keras/losses)
* Evaluierungsmetriken (https://www.tensorflow.org/api_docs/python/tf/keras/metrics)
* Optimizer (https://www.tensorflow.org/api_docs/python/tf/keras/optimizers) 

Nutzen Sie den zuvor erstellten Validierungsdatensatz für die Validierung des Modells während des Trainings.

### 4. Modelloptimierung
Optimieren Sie das Modell auf Grundlage der Validierungsergebnisse aus dem Trainingsvorgang. Passen Sie dafür nach und
nach sinnvoll einige der o.a. Konfigurationsparameter an und führen Sie ein erneutes Training durch. Wiederholen Sie 
diesen Schritt bis das Modell die gewünschte Leistung zeigt. Ziel sollte es sein, das optimale Gleichgewicht zwischen
Over- und Underfitting zu finden.

### 5. Evaluierung des Modells
Evaluieren Sie das Modell auf dem Testdatensatz für jeweils unterschiedliche Anzahl an n-vorangegangenen Tage und stellen
Sie ihre Ergebnisse sinnvoll in statistischen Diagrammen dar.
Ermitteln Sie hierzu auch das Nash-Sutcliff Effizienzkriterium (NSE) zu:  
<img src="https://render.githubusercontent.com/render/math?math=NSE=1-\frac{\sum_{i=1}^{N}(Q_{s,i}-Q_{o,i})^2}{\sum_{i=1}^{N}(Q_{o,i}-\overline{Q_{o}})^2}">  
<img src="https://render.githubusercontent.com/render/math?math=Q_{s,i}">: Simulierte/Vorhergesagte Abflusswerte  
<img src="https://render.githubusercontent.com/render/math?math=Q_{o,i}">: Beobachtete/Tatsächliche Abflusswerte  
<img src="https://render.githubusercontent.com/render/math?math=\overline{Q_{o}}">: Mittelwert der beobachteten Werte

Visualisieren Sie außerdem die tatsächlichen und die simulierten Abflusswerte für den Testdatensatz gemeinsam in einer Ganglinie.
Hierzu müssen Sie die simulierten Werte wieder in die ursprüngliche Maßeinheit zurückskalieren.

**Zusatzaufgabe**  
Implementieren Sie das NSE-Kriterium als Metrik in der Tensorflow API, so dass es direkt während des Trainings- und 
Validierungsvorgang so wie die bereits bekannten built-in Metriken berechnet werden kann. Hierzu ist eine eigene Klasse
zu implementieren, die `tf.keras.metrics.Metric` erweitert. Ein Beispiel hierzu finden Sie unter https://www.tensorflow.org/guide/keras/train_and_evaluate#custom_metrics

### 6. Erweiterungen (Optional)
1. Erweitern Sie ihre Implementierungen so, dass als Input Features nicht nur die Niederschlagswerte der n-vorangegangen Tage
verwendet werden können, sondern auch weitere meteorologische Parameter aus dem CAMEL-US Datensatzes als Input Features möglich sind.
2. Führen Sie die Modellierung auch für weitere Einzugsgebiete durch und stellen Sie die Ergebnisse in entsprechenden Diagrammen dar.
Nutzen Sie dafür einen festen Satz an Konfigurationsparametern für das Neuronale Netz, um die Ergebnisse der unterschiedlichen
Einzugsgebiete vergleichbar zu machen.
