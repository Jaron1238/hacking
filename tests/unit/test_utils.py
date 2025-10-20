

def test_lookup_vendor_apple_mac(self):
        """Test Vendor-Lookup für Apple-MAC."""
        # Apple MAC-Adresse
        mac = "a8:51:ab:0c:b9:e9"
        vendor = utils.lookup_vendor(mac)
        
        assert vendor is not None
        assert "Apple" in vendor

def test_lookup_vendor_randomized_mac(self):
        """Test Vendor-Lookup für randomisierte MAC."""
        # Randomisierte MAC-Adresse
        mac = "b2:87:23:15:7f:f2"
        vendor = utils.lookup_vendor(mac)
        
        assert vendor is not None
        assert "Randomisiert" in vendor

def test_lookup_vendor_unknown_mac(self):
        """Test Vendor-Lookup für unbekannte MAC."""
        # Unbekannte MAC-Adresse
        mac = "ff:ff:ff:ff:ff:ff"
        vendor = utils.lookup_vendor(mac)
        
        # Sollte None oder "Unknown" zurückgeben
        assert vendor is None or vendor == "Unknown"

def test_intelligent_vendor_lookup(self, mock_client_state):
        """Test intelligenter Vendor-Lookup."""
        # Mock ClientState mit Apple-IE
        mock_client_state.parsed_ies = {"vendor_specific": {"Apple": True}}
        
        vendor = utils.intelligent_vendor_lookup(mock_client_state.mac, mock_client_state)
        
        assert vendor is not None
        assert "Apple" in vendor

def test_intelligent_vendor_lookup_no_ies(self, mock_client_state):
        """Test intelligenter Vendor-Lookup ohne IEs."""
        # Mock ClientState ohne IEs
        mock_client_state.parsed_ies = {}
        
        vendor = utils.intelligent_vendor_lookup(mock_client_state.mac, mock_client_state)
        
        # Sollte auf Standard-Lookup zurückfallen
        assert vendor is not None

def test_download_oui_file_success(self, mock_oui_path):
        """Test OUI-Datei-Download (erfolgreich)."""
        # Mock temporäre Datei
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        mock_oui_path.return_value = Path(temp_path)
        
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b"test oui data"
            mock_urlopen.return_value.__enter__.return_value = mock_response
            
            result = utils.download_oui_file()
            
            assert result is True
            assert Path(temp_path).exists()
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

def test_download_oui_file_failure(self, mock_oui_path):
        """Test OUI-Datei-Download (Fehler)."""
        mock_oui_path.return_value = Path("/tmp/test_oui")
        
        with patch('urllib.request.urlopen', side_effect=Exception("Network error")):
            result = utils.download_oui_file()
            
            assert result is False