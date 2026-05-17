# Code zur Bachelorarbeit: Phononenkristalle in der Musik -- Halsabstimmung der E-Gitarre mittels lokaler Resonatoren, vorgelegt von Piet Kuchenbecker.


## Beschreibung

Dieses Repository enthält die für die Bachelorarbeit verwendeten Python-Skripte zur Auswertung und Analyse der Messdaten und zur Erstellung relevanter Grafiken.

In dieser Arbeit wurde eine E-Gitarre mit einem Metamaterial versehen und die Effekte dieser Struktur auf die Schwingung und den Klang der Gitarre analysiert.

Aufgrund der Datenmenge, liegen diesem Repository keine Rohdaten vor, sondern nur der verwendete Code.

Das Projekt wurde vorwiegend mit Python 3.12, in Teilen auch mit Python 3.14 entwickelt.


## Inhalt

# array_functions.py

Diese Datei beinhaltet die Kernfunktionen der Arraydatenanalyse.
Hierin stecken alle Funktionen zum Laden und Speichern von Datensets.
Außerdem wurden auf Grundlage dieser Datei die Metadaten erschaffen und die npz-Dateien zum Speichern der Array-Daten entwickelt.
Die Datei beinhaltet den Code für die grafischen Darstellungen der Array-Datensätze.
Finale Aufbereitungen der Datensets wurden mithilfe dieser Functions durchgeführt.
Um das versehentliche Löschen von Functions zu verhindern, wurde code teilweise nicht gelöscht, sondern in alternativen Varianten hinzugefügt.
Darunter leidet die Übersichtlichkeit, aber der Code bleibt stabil.
Es ist somit nicht klar, welche Functions nun wirklich alle verwendet wurden.

# run_array_functions.ipynb

Dies ist die Datei, mit der die Functions aus "array_functions.py" ausgeführt wurden.
Somit stecken hierin die gesamten Prozesse zur Analyse der Daten.
Mithilfe dieses interaktiven Skriptes lassen sich die Daten sehr einfach visualisieren.

# all_functions_array.ipynb

Diese Datei wurde hauptsächlich zum Erstellen der Datensets verwendet.
Sie beinhaltet das automatische Benennen der Dateien, das Speichern und Laden von Array-Datensätzen, die Onset-Detection und das finale Aufbereiten der Array-Dateien.
Die Datei entspringt dem frühen Entwicklungsprozess, weshalb einige Funktionen später überarbeitet in der Datei "array_functions.py" landeten, um übersichtlicher mithilfe der Datei "run_array_functions.ipynb" abgerufen werden zu können.

# pylotec_new.py

In dieser Datei stecken einige Functions, die für eine umfänglichere Auswertung ergänzend zur pylotec-Bibliothek geschrieben wurden.
Für ein Funktionsfähiges Skript müssen die Functions an das Ende der Datei pylotec.py hinzugefügt werden.
Sie beinhalten die Möglichkeit Modenkarten anstatt in Imaginär- und Realteilen als Phasen und Amplitudenkarten auszugeben.
Außerdem sind Visualisierungen zum Vergleich von Spektren in diesem Skript enthalten.

# Dispersionsrelation [...].ipynb

Die mit "Dispersionsrelation" anfangenden Jupyter-Notebooks beinhalten den Code, mit dem die entsprechenden Grafiken erstellt wurden.
Genauso ist ein Jupyter Notebook für die Erstellung der Geschwindigkeitsgrafik zu finden.

# Lautstärkeabweichungen.ipynb

Mithilfe dieser Datei wurde die Kalibrierung der Arraymikrofone in Testdurchläufen geprüft.

# Massenberechnung.ipynb

Einfaches Skript für die Berechnung der zur Verfügung stehenden Kugelmassen