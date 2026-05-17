<span style="color:Sienna; font-family:arial;font-size:12pt;">
    
It is important to define a **Python Virtual Environment** (PVE) for each Python project:<br> 
a PVE helps to control the versions of the Python interpreter and *sensitive modules* (tensorflow...)<br>
for each project.<br>

- a KAGGLE session can serve as a PVE,
- or if on a local machine, use the `uv` manager to create a PVE and run the notenbook with<br>
   `uv run jupyter lab` to ensure it uses the PVE of the projet.
</span>