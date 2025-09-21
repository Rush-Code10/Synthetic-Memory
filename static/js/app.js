/**
 * Synthetic Memory Lite - Main JavaScript Application
 * Handles all interactive features, AJAX calls, and UI animations
 */

class SyntheticMemoryApp {
    constructor() {
        this.isInitialized = false;
        this.currentQuery = '';
        this.currentResponse = null;
        this.ttsEnabled = false;
        this.selectedVoice = 'English (US) - Female';
        this.speechSpeed = 1.0;
        this.currentAudio = null;
        
        this.init();
    }

    async init() {
        try {
            // Show loading screen
            this.showLoadingScreen();
            
            // Initialize app
            await this.initializeApp();
            
            // Setup event listeners
            this.setupEventListeners();
            
            // Load initial data
            await this.loadAppStatus();
            
            // Hide loading screen
            this.hideLoadingScreen();
            
            this.isInitialized = true;
            console.log('Synthetic Memory Lite initialized successfully');
        } catch (error) {
            console.error('Initialization failed:', error);
            this.showError('Failed to initialize application. Please refresh the page.');
        }
    }

    showLoadingScreen() {
        const loadingScreen = document.getElementById('loading-screen');
        const mainApp = document.getElementById('main-app');
        
        if (loadingScreen) {
            loadingScreen.classList.remove('hidden');
        }
        if (mainApp) {
            mainApp.classList.add('hidden');
        }
    }

    hideLoadingScreen() {
        const loadingScreen = document.getElementById('loading-screen');
        const mainApp = document.getElementById('main-app');
        
        setTimeout(() => {
            if (loadingScreen) {
                loadingScreen.classList.add('fade-out');
                setTimeout(() => {
                    loadingScreen.classList.add('hidden');
                }, 500);
            }
            if (mainApp) {
                mainApp.classList.remove('hidden');
                mainApp.classList.add('fade-in-up');
            }
        }, 1000);
    }

    async initializeApp() {
        // Quick initialization without artificial delay
        await new Promise(resolve => setTimeout(resolve, 500));
    }

    setupEventListeners() {
        // Query input
        const queryInput = document.getElementById('query-input');
        if (queryInput) {
            queryInput.addEventListener('input', this.handleQueryInput.bind(this));
            queryInput.addEventListener('keypress', this.handleQueryKeypress.bind(this));
        }

        // Search button
        const searchBtn = document.getElementById('search-btn');
        if (searchBtn) {
            searchBtn.addEventListener('click', this.handleSearch.bind(this));
        }

        // Quick query buttons
        const quickQueries = document.querySelectorAll('.quick-query');
        quickQueries.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const query = e.target.getAttribute('data-query');
                this.setQuery(query);
            });
        });

        // Settings modal
        const settingsBtn = document.getElementById('settings-btn');
        if (settingsBtn) {
            settingsBtn.addEventListener('click', this.showSettingsModal.bind(this));
        }

        // Audio control buttons
        const audioPlayPause = document.getElementById('audio-play-pause');
        if (audioPlayPause) {
            audioPlayPause.addEventListener('click', this.toggleAudioPlayPause.bind(this));
        }

        const audioStop = document.getElementById('audio-stop');
        if (audioStop) {
            audioStop.addEventListener('click', this.stopAudio.bind(this));
        }

        const audioClose = document.getElementById('audio-close');
        if (audioClose) {
            audioClose.addEventListener('click', this.hideInlineAudioPlayer.bind(this));
        }

        // TTS controls in settings
        const voiceSelect = document.getElementById('voice-select');
        if (voiceSelect) {
            voiceSelect.addEventListener('change', this.handleVoiceChange.bind(this));
        }

        const speedSlider = document.getElementById('speed-slider');
        if (speedSlider) {
            speedSlider.addEventListener('input', this.handleSpeedChange.bind(this));
        }

        const previewBtn = document.getElementById('preview-voice');
        if (previewBtn) {
            previewBtn.addEventListener('click', this.previewVoice.bind(this));
        }

        // Speak results button
        const speakResultsBtn = document.getElementById('speak-results');
        if (speakResultsBtn) {
            speakResultsBtn.addEventListener('click', this.speakResults.bind(this));
        }

        // Clear results
        const clearBtn = document.getElementById('clear-results');
        if (clearBtn) {
            clearBtn.addEventListener('click', this.clearResults.bind(this));
        }

        // Toggle agent panel
        const toggleAgent = document.getElementById('toggle-agent');
        if (toggleAgent) {
            toggleAgent.addEventListener('click', this.toggleAgentPanel.bind(this));
        }

        // Modal controls
        const modalCloses = document.querySelectorAll('.modal-close');
        modalCloses.forEach(btn => {
            btn.addEventListener('click', this.closeModal.bind(this));
        });

        // Toast close buttons
        const toastCloses = document.querySelectorAll('.toast-close');
        toastCloses.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.target.closest('.toast').classList.remove('show');
            });
        });

        // Click outside modals to close
        const modals = document.querySelectorAll('.modal');
        modals.forEach(modal => {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    this.closeModal();
                }
            });
        });
    }

    async loadAppStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            if (data.status === 'ready') {
                this.loadVoices();
            } else {
                console.error('App not ready:', data);
            }
        } catch (error) {
            console.error('Failed to load app status:', error);
        }
    }

    async loadVoices() {
        try {
            const response = await fetch('/api/voices');
            const voices = await response.json();
            
            const voiceSelect = document.getElementById('voice-select');
            if (voiceSelect) {
                voiceSelect.innerHTML = '';
                Object.keys(voices).forEach(voiceName => {
                    const option = document.createElement('option');
                    option.value = voiceName;
                    option.textContent = voiceName;
                    voiceSelect.appendChild(option);
                });
            }
        } catch (error) {
            console.error('Failed to load voices:', error);
        }
    }

    // Removed updateStatusIndicator and updateStats functions as they're no longer needed

    handleQueryInput(e) {
        const query = e.target.value;
        this.currentQuery = query;
        
        // Update character count
        const charCount = document.getElementById('char-count');
        if (charCount) {
            charCount.textContent = `${query.length}/500 characters`;
            
            if (query.length > 400) {
                charCount.style.color = '#f59e0b';
            } else if (query.length > 0) {
                charCount.style.color = '#64748b';
            }
        }
    }

    handleQueryKeypress(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.handleSearch();
        }
    }

    setQuery(query) {
        const queryInput = document.getElementById('query-input');
        if (queryInput) {
            queryInput.value = query;
            this.currentQuery = query;
            this.handleQueryInput({ target: { value: query } });
        }
    }

    async handleSearch() {
        if (!this.currentQuery.trim()) {
            this.showError('Please enter a query');
            return;
        }

        try {
            // Update UI
            this.setSearchButtonLoading(true);
            this.clearResults();
            
            // Show AI thinking panel
            this.showAIThinkingPanel();

            // Make API call
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: this.currentQuery,
                    tts_enabled: this.ttsEnabled,
                    voice: this.selectedVoice
                })
            });

            const data = await response.json();

            if (data.success) {
                this.currentResponse = data;
                this.displayResults(data);
                
                // Show speak results button
                this.showSpeakResultsButton();
                
                // Update AI thinking panel with results
                this.updateAIThinkingPanel(data);
                
                this.showSuccess('Analysis completed successfully');
            } else {
                this.showError(data.error || 'Search failed');
            }
        } catch (error) {
            console.error('Search error:', error);
            this.showError('Analysis failed. Please try again.');
        } finally {
            // Query completed
            this.setSearchButtonLoading(false);
        }
    }

    setSearchButtonLoading(loading) {
        const searchBtn = document.getElementById('search-btn');
        if (searchBtn) {
            if (loading) {
                searchBtn.classList.add('loading');
                searchBtn.disabled = true;
                // Keep original content but make it invisible - loading wheel will show via CSS
            } else {
                searchBtn.classList.remove('loading');
                searchBtn.disabled = false;
                searchBtn.innerHTML = '<span class="btn-text">Search</span><i class="fas fa-arrow-right"></i>';
            }
        }
    }

    displayResults(data) {
        const resultsContent = document.getElementById('results-content');
        if (!resultsContent) return;

        const response = data.response;
        
        resultsContent.innerHTML = `
            <div class="result-answer fade-in-up">
                <h4>Analysis Results</h4>
                <div class="result-text">${this.formatAnswerText(response.answer)}</div>
            </div>
            
            ${response.sources && response.sources.length > 0 ? `
                <div class="result-sources fade-in-up">
                    <h5>Source References</h5>
                    <div class="source-list">
                        ${response.sources.map((source, index) => this.formatSourceItem(source, index + 1)).join('')}
                    </div>
                </div>
            ` : ''}
            
            <div class="result-stats fade-in-up">
                <div class="stats-grid">
                    <div class="stat-item">
                        <i class="fas fa-search"></i>
                        <span>${data.search_terms.length} search terms used</span>
                    </div>
                    <div class="stat-item">
                        <i class="fas fa-envelope"></i>
                        <span>${data.stats.email_results} emails analyzed</span>
                    </div>
                    <div class="stat-item">
                        <i class="fab fa-slack"></i>
                        <span>${data.stats.slack_results} messages reviewed</span>
                    </div>
                    <div class="stat-item">
                        <i class="fas fa-file-alt"></i>
                        <span>${data.stats.document_found ? 'Document content found' : 'No document matches'}</span>
                    </div>
                </div>
            </div>
        `;

        // Show results section
        const resultsSection = document.getElementById('results-section');
        if (resultsSection) {
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        }
    }

    formatAnswerText(text) {
        if (!text) return '';
        
        // Convert superscript citations [1] to HTML
        text = text.replace(/\^(\d+)\^/g, '<sup>$1</sup>');
        
        // Convert bullet points
        text = text.replace(/^•\s*/gm, '<span class="bullet">•</span> ');
        
        // Convert line breaks
        text = text.replace(/\n/g, '<br>');
        
        return text;
    }

    formatSourceItem(source, index) {
        const sourceMapping = this.currentResponse?.response?.source_mapping || {};
        const sourceData = sourceMapping[source.id] || {};
        
        return `
            <div class="source-item slide-in-right" style="animation-delay: ${index * 0.1}s">
                <div class="source-header">
                    <span class="source-type">${source.type}</span>
                    <span class="source-meta">${sourceData.description || 'Source information'}</span>
                </div>
                <div class="source-content">${source.content_snippet || 'No content preview available'}</div>
            </div>
        `;
    }

    async generateTTS(text) {
        if (!text || !this.ttsEnabled) return;

        try {
            const response = await fetch('/api/tts', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    voice: this.selectedVoice
                })
            });

            const data = await response.json();

            if (data.success) {
                this.showAudioModal(data.audio_data);
            } else {
                console.error('TTS generation failed:', data.error);
            }
        } catch (error) {
            console.error('TTS error:', error);
        }
    }

    showAudioModal(audioData) {
        const modal = document.getElementById('audio-modal');
        const audioPlayer = document.getElementById('audio-player');
        
        if (modal && audioPlayer) {
            audioPlayer.src = audioData;
            modal.classList.add('active');
            
            // Store audio data for download
            this.currentAudioData = audioData;
        }
    }

    closeModal() {
        const modals = document.querySelectorAll('.modal');
        modals.forEach(modal => {
            modal.classList.remove('active');
        });
    }

    showSettingsModal() {
        const modal = document.getElementById('settings-modal');
        if (modal) {
            modal.classList.add('active');
        }
    }

    showSpeakResultsButton() {
        const speakBtn = document.getElementById('speak-results');
        if (speakBtn) {
            speakBtn.style.display = 'inline-flex';
            speakBtn.classList.add('show');
        }
    }

    hideSpeakResultsButton() {
        const speakBtn = document.getElementById('speak-results');
        if (speakBtn) {
            speakBtn.style.display = 'none';
            speakBtn.classList.remove('show');
        }
    }

    async speakResults() {
        if (!this.currentResponse || !this.currentResponse.response.answer) {
            this.showError('No results to speak');
            return;
        }

        // Show loading state
        this.showTTSLoading();

        try {
            const response = await fetch('/api/tts', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: this.currentResponse.response.answer,
                    voice: this.selectedVoice
                })
            });

            const data = await response.json();

            if (data.success) {
                this.hideTTSLoading();
                this.showInlineAudioPlayer(data.audio_data);
            } else {
                this.hideTTSLoading();
                this.showError('Text-to-speech generation failed');
            }
        } catch (error) {
            console.error('TTS error:', error);
            this.hideTTSLoading();
            this.showError('Text-to-speech generation failed');
        }
    }

    toggleTTS() {
        const ttsPanel = document.getElementById('tts-panel');
        const ttsContent = document.querySelector('.tts-content');
        const ttsToggle = document.getElementById('tts-toggle');
        
        this.ttsEnabled = !this.ttsEnabled;
        
        if (ttsContent) {
            ttsContent.classList.toggle('active', this.ttsEnabled);
        }
        if (ttsToggle) {
            ttsToggle.classList.toggle('active', this.ttsEnabled);
        }
    }

    handleVoiceChange(e) {
        this.selectedVoice = e.target.value;
    }

    handleSpeedChange(e) {
        this.speechSpeed = parseFloat(e.target.value);
        const speedValue = document.getElementById('speed-value');
        if (speedValue) {
            speedValue.textContent = `${this.speechSpeed}x`;
        }
    }

    async previewVoice() {
        try {
            const response = await fetch('/api/tts/preview', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    voice: this.selectedVoice
                })
            });

            const data = await response.json();

            if (data.success) {
                this.showAudioModal(data.audio_data);
            } else {
                this.showError('Voice preview failed');
            }
        } catch (error) {
            console.error('Voice preview error:', error);
            this.showError('Voice preview failed');
        }
    }

    showTTSLoading() {
        const speakBtn = document.getElementById('speak-results');
        if (speakBtn) {
            speakBtn.innerHTML = '<span class="tts-loading"><span class="tts-loading-text">Generating audio...</span></span>';
            speakBtn.disabled = true;
        }
    }

    hideTTSLoading() {
        const speakBtn = document.getElementById('speak-results');
        if (speakBtn) {
            speakBtn.innerHTML = '<i class="fas fa-volume-up"></i>Speak Results';
            speakBtn.disabled = false;
        }
    }

    showInlineAudioPlayer(audioData) {
        const inlinePlayer = document.getElementById('inline-audio-player');
        const audioStatus = document.querySelector('.audio-status');
        const statusText = document.querySelector('.status-text');
        const audioLoading = document.querySelector('.audio-loading');
        
        if (inlinePlayer && audioStatus) {
            // Create hidden audio element
            if (!this.currentAudio) {
                this.currentAudio = new Audio();
                this.setupAudioEventListeners();
            }
            
            this.currentAudio.src = audioData;
            this.currentAudio.load();
            
            // Show the inline player
            inlinePlayer.style.display = 'block';
            statusText.textContent = 'Ready to play...';
            audioLoading.style.display = 'none';
            
            // Auto-play
            this.currentAudio.play().then(() => {
                statusText.textContent = 'Playing audio response...';
                this.updatePlayPauseButton(true);
            }).catch(error => {
                console.error('Auto-play failed:', error);
                statusText.textContent = 'Click play to start audio';
                this.updatePlayPauseButton(false);
            });
        }
    }

    hideInlineAudioPlayer() {
        const inlinePlayer = document.getElementById('inline-audio-player');
        if (inlinePlayer) {
            inlinePlayer.style.display = 'none';
        }
        
        if (this.currentAudio) {
            this.currentAudio.pause();
            this.currentAudio.currentTime = 0;
        }
    }

    setupAudioEventListeners() {
        if (!this.currentAudio) return;

        this.currentAudio.addEventListener('timeupdate', () => {
            this.updateProgress();
        });

        this.currentAudio.addEventListener('ended', () => {
            this.hideInlineAudioPlayer();
        });

        this.currentAudio.addEventListener('pause', () => {
            this.updatePlayPauseButton(false);
            const statusText = document.querySelector('.status-text');
            if (statusText) {
                statusText.textContent = 'Paused';
            }
        });

        this.currentAudio.addEventListener('play', () => {
            this.updatePlayPauseButton(true);
            const statusText = document.querySelector('.status-text');
            if (statusText) {
                statusText.textContent = 'Playing audio response...';
            }
        });
    }

    updatePlayPauseButton(isPlaying) {
        const playPauseBtn = document.getElementById('audio-play-pause');
        if (playPauseBtn) {
            const icon = playPauseBtn.querySelector('i');
            if (icon) {
                icon.className = isPlaying ? 'fas fa-pause' : 'fas fa-play';
            }
        }
    }

    updateProgress() {
        if (!this.currentAudio) return;
        
        const progressFill = document.querySelector('.audio-progress-fill');
        const currentTimeEl = document.querySelector('.current-time');
        const totalTimeEl = document.querySelector('.total-time');
        
        if (progressFill && this.currentAudio.duration > 0) {
            const progress = (this.currentAudio.currentTime / this.currentAudio.duration) * 100;
            progressFill.style.width = `${progress}%`;
        }
        
        if (currentTimeEl) {
            currentTimeEl.textContent = this.formatTime(this.currentAudio.currentTime);
        }
        
        if (totalTimeEl) {
            totalTimeEl.textContent = this.formatTime(this.currentAudio.duration);
        }
    }

    formatTime(seconds) {
        if (isNaN(seconds)) return '0:00';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    toggleAudioPlayPause() {
        if (!this.currentAudio) return;
        
        if (this.currentAudio.paused) {
            this.currentAudio.play();
        } else {
            this.currentAudio.pause();
        }
    }

    stopAudio() {
        if (this.currentAudio) {
            this.currentAudio.pause();
            this.currentAudio.currentTime = 0;
            this.updateProgress();
        }
    }

    showAIThinkingPanel() {
        const agentContent = document.getElementById('agent-content');
        const toggleBtn = document.getElementById('toggle-agent');
        
        if (agentContent && toggleBtn) {
            agentContent.classList.add('active');
            toggleBtn.innerHTML = '<i class="fas fa-eye-slash"></i> Hide Details';
            this.loadAgentThoughts();
        }
    }

    updateAIThinkingPanel(data) {
        const thinkingSteps = document.getElementById('thinking-steps');
        if (thinkingSteps) {
            thinkingSteps.innerHTML = `
                <div class="thinking-step">
                    <div class="step-header">
                        <div class="step-number">1</div>
                        <div class="step-title">Query Analysis</div>
                    </div>
                    <div class="step-description">Analyzed user query and generated search terms</div>
                    <div class="step-details">Query: "${this.currentQuery}"\nGenerated ${data.search_terms ? data.search_terms.length : 0} search terms for comprehensive data retrieval</div>
                    <div class="step-status completed">
                        <i class="fas fa-check"></i>
                        Completed
                    </div>
                </div>
                
                <div class="thinking-step">
                    <div class="step-header">
                        <div class="step-number">2</div>
                        <div class="step-title">Data Search</div>
                    </div>
                    <div class="step-description">Searched through emails, messages, and documents</div>
                    <div class="step-details">Found ${data.stats ? data.stats.email_results : 0} email matches, ${data.stats ? data.stats.slack_results : 0} message matches, ${data.stats && data.stats.document_found ? '1' : '0'} document matches</div>
                    <div class="step-status completed">
                        <i class="fas fa-check"></i>
                        Completed
                    </div>
                </div>
                
                <div class="thinking-step">
                    <div class="step-header">
                        <div class="step-number">3</div>
                        <div class="step-title">AI Synthesis</div>
                    </div>
                    <div class="step-description">Used Gemini AI to synthesize findings into coherent response</div>
                    <div class="step-details">Processed ${data.stats ? (data.stats.email_results + data.stats.slack_results + (data.stats.document_found ? 1 : 0)) : 0} data sources and generated AI-powered analysis with proper source attribution</div>
                    <div class="step-status completed">
                        <i class="fas fa-check"></i>
                        Completed
                    </div>
                </div>
                
                <div class="thinking-step">
                    <div class="step-header">
                        <div class="step-number">4</div>
                        <div class="step-title">Response Generation</div>
                    </div>
                    <div class="step-description">Formatted final response with citations and sources</div>
                    <div class="step-details">Generated comprehensive response with ${data.response && data.response.sources ? data.response.sources.length : 0} source references and proper formatting</div>
                    <div class="step-status completed">
                        <i class="fas fa-check"></i>
                        Completed
                    </div>
                </div>
            `;
        }
    }

    clearResults() {
        const resultsContent = document.getElementById('results-content');
        if (resultsContent) {
            resultsContent.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-search"></i>
                    <h4>Ready for Analysis</h4>
                    <p>Enter your question above to get AI-powered insights from your data.</p>
                </div>
            `;
        }
        
        this.hideSpeakResultsButton();
        this.hideInlineAudioPlayer();
        this.currentResponse = null;
    }

    toggleAgentPanel() {
        const agentContent = document.getElementById('agent-content');
        const toggleBtn = document.getElementById('toggle-agent');
        
        if (agentContent && toggleBtn) {
            const isActive = agentContent.classList.contains('active');
            
            agentContent.classList.toggle('active', !isActive);
            
            if (!isActive) {
                toggleBtn.innerHTML = '<i class="fas fa-eye-slash"></i> Hide Details';
                this.loadAgentThoughts();
            } else {
                toggleBtn.innerHTML = '<i class="fas fa-eye"></i> Show Details';
            }
        }
    }

    async loadAgentThoughts() {
        try {
            const response = await fetch('/api/agent-thoughts');
            const data = await response.json();
            
            const thinkingSteps = document.getElementById('thinking-steps');
            if (thinkingSteps) {
                thinkingSteps.innerHTML = `
                    <div class="thinking-step">
                        <div class="step-header">
                            <div class="step-number">1</div>
                            <div class="step-title">Query Analysis</div>
                        </div>
                        <div class="step-description">Analyzing user query and generating search terms</div>
                        <div class="step-details">${data.query_analysis || `Analyzed query: "${this.currentQuery}" and generated search terms`}</div>
                        <div class="step-status ${data.query_analysis ? 'completed' : 'processing'}">
                            <i class="fas ${data.query_analysis ? 'fa-check' : 'fa-spinner'}"></i>
                            ${data.query_analysis ? 'Completed' : 'Processing...'}
                        </div>
                    </div>
                    
                    <div class="thinking-step">
                        <div class="step-header">
                            <div class="step-number">2</div>
                            <div class="step-title">Data Search</div>
                        </div>
                        <div class="step-description">Searching through emails, messages, and documents</div>
                        <div class="step-details">${data.search_results || 'Searching across emails, Slack messages, and documents for relevant content'}</div>
                        <div class="step-status ${data.search_results ? 'completed' : 'processing'}">
                            <i class="fas ${data.search_results ? 'fa-check' : 'fa-spinner'}"></i>
                            ${data.search_results ? 'Completed' : 'Processing...'}
                        </div>
                    </div>
                    
                    <div class="thinking-step">
                        <div class="step-header">
                            <div class="step-number">3</div>
                            <div class="step-title">AI Synthesis</div>
                        </div>
                        <div class="step-description">Using Gemini AI to synthesize findings into coherent response</div>
                        <div class="step-details">${data.synthesis_process || 'Combining findings using AI analysis with proper source attribution'}</div>
                        <div class="step-status ${data.synthesis_process ? 'completed' : 'processing'}">
                            <i class="fas ${data.synthesis_process ? 'fa-check' : 'fa-spinner'}"></i>
                            ${data.synthesis_process ? 'Completed' : 'Processing...'}
                        </div>
                    </div>
                    
                    <div class="thinking-step">
                        <div class="step-header">
                            <div class="step-number">4</div>
                            <div class="step-title">Response Generation</div>
                        </div>
                        <div class="step-description">Formatting final response with citations and sources</div>
                        <div class="step-details">${data.response_generation || 'Generating final response with proper formatting and source references'}</div>
                        <div class="step-status ${data.response_generation ? 'completed' : 'processing'}">
                            <i class="fas ${data.response_generation ? 'fa-check' : 'fa-spinner'}"></i>
                            ${data.response_generation ? 'Completed' : 'Processing...'}
                        </div>
                    </div>
                `;
            }
        } catch (error) {
            console.error('Failed to load agent thoughts:', error);
            const thinkingSteps = document.getElementById('thinking-steps');
            if (thinkingSteps) {
                thinkingSteps.innerHTML = `
                    <div class="thinking-step">
                        <div class="step-header">
                            <div class="step-number">!</div>
                            <div class="step-title">Error Loading Details</div>
                        </div>
                        <div class="step-description">Unable to load AI thinking process details</div>
                        <div class="step-status error">
                            <i class="fas fa-exclamation-triangle"></i>
                            Error
                        </div>
                    </div>
                `;
            }
        }
    }

    showError(message) {
        this.showToast('error', message);
    }

    showSuccess(message) {
        this.showToast('success', message);
    }

    showToast(type, message) {
        const toast = document.getElementById(`${type}-toast`);
        const messageEl = document.getElementById(`${type}-message`);
        
        if (toast && messageEl) {
            messageEl.textContent = message;
            toast.classList.add('show');
            
            // Auto hide after 5 seconds
            setTimeout(() => {
                toast.classList.remove('show');
            }, 5000);
        }
    }
}

// Download audio functionality
document.addEventListener('click', (e) => {
    if (e.target.id === 'download-audio' && window.app?.currentAudioData) {
        const audioData = window.app.currentAudioData;
        const link = document.createElement('a');
        link.href = audioData;
        link.download = 'synthetic_memory_response.mp3';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
});

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new SyntheticMemoryApp();
});

// Add some additional CSS for the new elements
const additionalCSS = `
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .stat-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem;
        background: var(--gray-50);
        border-radius: var(--radius-md);
        font-size: var(--font-size-sm);
        color: var(--gray-700);
    }
    
    .stat-item i {
        color: var(--primary-color);
    }
    
    .bullet {
        color: var(--primary-color);
        font-weight: 600;
    }
    
    .status-indicator.ready .status-dot {
        background: var(--success-color);
    }
    
    .status-indicator.error .status-dot {
        background: var(--error-color);
    }
    
    .status-indicator.loading .status-dot {
        background: var(--warning-color);
    }
`;

// Inject additional CSS
const style = document.createElement('style');
style.textContent = additionalCSS;
document.head.appendChild(style);
