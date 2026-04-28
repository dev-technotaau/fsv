"""
Simple HTTP Server for SegFormer Web Interface
===============================================
Serves the HTML file and ONNX model for local testing.

Usage:
    python start_web_server.py
    
Then open: http://localhost:8000
"""

import http.server
import socketserver
import os
from pathlib import Path

# Configuration
PORT = 8080  # Changed from 8000 (try alternative ports if blocked)
ALTERNATIVE_PORTS = [8080, 8888, 3000, 5000, 8001, 9000]  # Fallback ports
DIRECTORY = Path(__file__).resolve().parent.parent.parent  # Serve from project root

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler with CORS and proper MIME types"""
    
    def end_headers(self):
        # Enable CORS for ONNX model loading
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        
        # Cache control
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        
        super().end_headers()
    
    def guess_type(self, path):
        """Add ONNX MIME type"""
        mimetype = super().guess_type(path)
        if path.endswith('.onnx'):
            return 'application/octet-stream'
        return mimetype
    
    def do_OPTIONS(self):
        """Handle preflight requests"""
        self.send_response(200)
        self.end_headers()


def main():
    """Start the web server"""
    
    # Change to directory
    os.chdir(DIRECTORY)
    
    print("=" * 70)
    print("SEGFORMER WEB SERVER")
    print("=" * 70)
    
    # Try to find available port
    selected_port = None
    for port in ALTERNATIVE_PORTS:
        try:
            # Test if port is available
            test_socket = socketserver.TCPServer(("", port), MyHTTPRequestHandler)
            test_socket.server_close()
            selected_port = port
            break
        except OSError:
            continue
    
    if selected_port is None:
        print("✗ Error: No available ports found!")
        print(f"  Tried ports: {ALTERNATIVE_PORTS}")
        print("\nSolutions:")
        print("  1. Close other applications using these ports")
        print("  2. Run as Administrator")
        print("  3. Try: netstat -ano | findstr :8080")
        return
    
    print(f"✓ Found available port: {selected_port}")
    print(f"Serving from: {DIRECTORY}")
    print()
    print("📂 Available pages (from project root):")
    print("   ✓ web/segformer/index_segformer_web.html  (main SegFormer demo)")
    print("   ✓ web/unet/index.html                      (UNet demo)")
    print("   ✓ web/unet_plusplus/index_unet_plusplus.html")
    print("   ✓ onnx_models/segformer_fence_detector.onnx")
    print("   ✓ models/onnx/*.onnx")
    print()
    print(f"🌐 Open in browser: http://localhost:{selected_port}/web/segformer/index_segformer_web.html")
    print()
    print("Press Ctrl+C to stop server")
    print("=" * 70)
    
    try:
        with socketserver.TCPServer(("", selected_port), MyHTTPRequestHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped")
    except Exception as e:
        print(f"\n✗ Server error: {e}")
        print("\nTroubleshooting:")
        print("  - Try running as Administrator")
        print("  - Check Windows Firewall settings")
        print("  - Close other applications using the port")


if __name__ == "__main__":
    main()
