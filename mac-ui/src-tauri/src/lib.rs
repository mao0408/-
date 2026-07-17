use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Manager, RunEvent};
use tauri_plugin_shell::ShellExt;

struct BackendState {
    child: Mutex<Option<tauri_plugin_shell::process::CommandChild>>,
}

fn start_backend(app: &AppHandle, state: &BackendState) {
    let resource_name = if cfg!(windows) { "backend.exe" } else { "backend" };
    let resource_path = app
        .path()
        .resolve(resource_name, tauri::path::BaseDirectory::Resource)
        .expect("resource path");

    if !resource_path.exists() {
        eprintln!("[backend] not found at {:?}", resource_path);
        return;
    }

    match app
        .shell()
        .command(resource_path.to_string_lossy().to_string())
        .env("BACKEND_PORT", "8765")
        .spawn()
    {
        Ok((_, child)) => {
            *state.child.lock().expect("backend state") = Some(child);
        }
        Err(error) => eprintln!("[backend] failed to start: {error}"),
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let state = Arc::new(BackendState {
        child: Mutex::new(None),
    });
    let setup_state = Arc::clone(&state);

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(Arc::clone(&state))
        .setup(move |app| {
            start_backend(app.handle(), &setup_state);
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building Tauri application")
        .run(move |_app, event| {
            if let RunEvent::Exit = event {
                if let Some(child) = state.child.lock().expect("backend state").take() {
                    let _ = child.kill();
                }
            }
        });
}
