"""
YOLO Layer Analyzer - Minimal version for testing
"""

def analyze_yolo11_layers(model):
    """Analyze YOLOv11 model layers"""
    total_params = sum(p.numel() for p in model.parameters())
    return {
        'total_params': total_params,
        'backbone_layers': [],
        'head_layers': []
    }

def get_two_phase_schedule(total_epochs, phase1_epochs):
    """Get two-phase training schedule"""
    schedule = []
    for epoch in range(total_epochs):
        phase = 'head-only' if epoch < phase1_epochs else 'full'
        schedule.append((epoch, phase))
    return schedule