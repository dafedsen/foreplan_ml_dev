def update_process_status_progress(id_prj, id_version):
    query = text("SELECT running_model FROM fplan_prj_prc WHERE id_prj = :id_prj AND id_version = :id_version")
    params = {
        "id_prj": int(id_prj),
        "id_version": int(id_version),
    }
    
    with engine.connect() as conn:
        result = conn.execute(query, params)
        result = result.fetchone()
        running_models = result[0]

    query = text("SELECT model_finished FROM fplan_prj_prc WHERE id_prj = :id_prj AND id_version = :id_version")
    params = {
        "id_prj": int(id_prj),
        "id_version": int(id_version),
    }
    
    with engine.connect() as conn:
        result = conn.execute(query, params)
        result = result.fetchone()
        model_finished = result[0]

    status_progress = math.ceil((model_finished / running_models) * 100)

    query = text("UPDATE fplan_prj_prc SET status_progress = :status_progress WHERE id_prj = :id_prj AND id_version = :id_version")
    params = {
        "id_prj": int(id_prj),
        "id_version": int(id_version),
        "status_progress": int(status_progress)
    }
    
    with engine.connect() as conn:
        conn.execute(query, params)
        conn.commit()

    engine.dispose()