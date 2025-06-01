// script.js
document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const turingScoreEl = document.getElementById('turingScore');
    const scherbiusScoreEl = document.getElementById('scherbiusScore');
    const maxVictoryPointsEl = document.getElementById('maxVictoryPoints');
    const winnerEl = document.getElementById('winner');
    const playAsTuringBtn = document.getElementById('playAsTuringBtn');
    const playAsScherbiusBtn = document.getElementById('playAsScherbiusBtn');

    // Player-specific controls and elements
    const turingControlsEl = document.getElementById('turingControls');
    const turingHandAreaEl = document.getElementById('turingHandArea'); // Area containing title and hand
    const turingHandEl = document.getElementById('turingHand');         // Actual card container
    const turingSubmitActionBtn = document.getElementById('turingSubmitActionBtn');

    const scherbiusControlsEl = document.getElementById('scherbiusControls');
    const scherbiusHandAreaEl = document.getElementById('scherbiusHandArea'); // Area containing title and hand
    const scherbiusHandEl = document.getElementById('scherbiusHand');         // Actual card container
    const scherbiusSubmitActionBtn = document.getElementById('scherbiusSubmitActionBtn');
    const scherbiusEncryptCheckbox = document.getElementById('scherbiusEncryptCheckbox');

    const gameAreaEl = document.getElementById('gameArea');
    const gameOverMessageEl = document.getElementById('gameOverMessage');
    const rewardsDisplayEl = document.getElementById('rewardsDisplay'); // Battle items will go here
    const battleZoneTitleEl = document.getElementById('battleZoneTitle');

    // --- DOM Elements for Historical View ---
    const historicalRoundViewAreaEl = document.getElementById('historicalRoundViewArea');
    const prevRoundBtnEl = document.getElementById('prevRoundBtn');
    const nextRoundBtnEl = document.getElementById('nextRoundBtn');
    const historicalRoundIndicatorEl = document.getElementById('historicalRoundIndicator');

    // Helper function to get current player's hand DOM element
    function getCurrentPlayerHandEl() {
        if (!clientState.playerRole) return null;
        return clientState.playerRole === 'Turing' ? turingHandEl : scherbiusHandEl;
    }

    // --- Client-Side Game State ---
    let clientState = {
        playerRole: null, // "Turing" or "Scherbius"
        playerHandForTurn: [], // Player's current hand objects
        currentHandDisplayObjects: [], // Cards currently shown in hand (after some are played)
        battlePlays: {}, // Player's card placements for battles: { "battle_0": [cardObj1], ... }
        nBattles: 0,
        maxCardsPerBattle: 0,
        draggedCard: { id: null, value: null, originType: null, originId: null, element: null, originSlotIndex: null },
        roundHistory: [],
        currentHistoryViewIndex: 0,
        latestServerState: null,
    };
    const VIEWING_CURRENT_ROUND_INDEX = () => clientState.roundHistory.length;

    // --- API Calls ---
    async function fetchApi(endpoint, method = 'GET', body = null) {
        try {
            const options = { method };
            if (body) {
                options.headers = { 'Content-Type': 'application/json' };
                options.body = JSON.stringify(body);
            }
            const response = await fetch(endpoint, options);
            if (!response.ok) {
                const errorData = await response.json();
                console.error(`API Error (${endpoint}):`, errorData.error || `HTTP error! Status: ${response.status}`);
                alert(`Error: ${errorData.error || `HTTP error! Status: ${response.status}`}`);
                if (response.status === 404 && endpoint === '/game_state') {
                    gameAreaEl.style.display = 'none';
                    document.querySelector('.game-controls').style.display = 'block';
                }
                return;
            }
            const state = await response.json();
            updateGlobalUI(state, endpoint);
        } catch (error) {
            console.error(`API Error (${endpoint}):`, error);
            alert(`Error: ${error.message}`);
        }
    }

    // --- Main UI Update Function ---
    function updateGlobalUI(serverState, endpointCalled) {
        console.log("Server state received:", serverState, "from endpoint:", endpointCalled);
        clientState.latestServerState = serverState;

        document.querySelector('.game-controls').style.display = 'none';
        gameAreaEl.style.display = 'block';

        clientState.playerRole = serverState.player_role;

        if (serverState.player_role === 'Turing') {
            turingScoreEl.textContent = serverState.turing_points;
            scherbiusScoreEl.textContent = '???';
        } else { // Scherbius or Observer (if implemented)
            turingScoreEl.textContent = '???';
            scherbiusScoreEl.textContent = serverState.scherbius_points;
        }
        maxVictoryPointsEl.textContent = serverState.max_victory_points;

        clientState.nBattles = serverState.n_battles;
        clientState.maxCardsPerBattle = serverState.max_cards_per_battle;
        clientState.roundHistory = serverState.round_history || [];

        const newRoundHistoryLength = clientState.roundHistory.length;

        if (serverState.is_game_over) {
            if (newRoundHistoryLength > 0) {
                clientState.currentHistoryViewIndex = newRoundHistoryLength - 1;
            } else { 
                clientState.currentHistoryViewIndex = VIEWING_CURRENT_ROUND_INDEX(); 
            }
        } else {
            if (endpointCalled === '/submit_player_action' && newRoundHistoryLength > 0) {
                clientState.currentHistoryViewIndex = newRoundHistoryLength - 1;
            } else { 
                clientState.currentHistoryViewIndex = VIEWING_CURRENT_ROUND_INDEX();
            }
        }

        const isPlayerActionPhase = (clientState.playerRole === "Turing" && serverState.current_phase === "Turing_Action") ||
                                    (clientState.playerRole === "Scherbius" && serverState.current_phase === "Scherbius_Action");

        if (isPlayerActionPhase && clientState.currentHistoryViewIndex === VIEWING_CURRENT_ROUND_INDEX()) {
            clientState.playerHandForTurn = [...(serverState.player_hand || [])];
            clientState.battlePlays = {};
            for (let i = 0; i < clientState.nBattles; i++) clientState.battlePlays[`battle_${i}`] = [];
            if (clientState.playerRole === 'Scherbius') {
                scherbiusEncryptCheckbox.checked = false;
            }
        }

        renderMainBattleDisplay(serverState);
        historicalRoundViewAreaEl.style.display = 'block';
        updateHistoryNavigationControls();
        manageRoundViewSpecificUI();

        if (serverState.is_game_over) {
            gameOverMessageEl.style.display = 'block';
            winnerEl.textContent = serverState.winner;
        } else {
            gameOverMessageEl.style.display = 'none';
        }
    }

    // --- Card Rendering and D&D ---
    function createCardElement(cardObject, originType, originId, originSlotIndex = null, playerContext = null) {
        const cardDiv = document.createElement('div');
        cardDiv.classList.add('card');
        cardDiv.textContent = cardObject.value;
        cardDiv.draggable = true;
        cardDiv.dataset.cardId = cardObject.id;
        cardDiv.dataset.cardValue = cardObject.value;
        cardDiv.dataset.originType = originType;
        cardDiv.dataset.originId = originId;
        if (originSlotIndex !== null) {
            cardDiv.dataset.originSlotIndex = originSlotIndex;
        }

        // Add player-specific class for styling based on who owns/plays the card
        if (playerContext === 'Turing') {
            cardDiv.classList.add('turing-card-ingame');
        } else if (playerContext === 'Scherbius') {
            cardDiv.classList.add('scherbius-card-ingame');
        }


        cardDiv.addEventListener('dragstart', (e) => {
            const isActivePlayerTurn = (clientState.playerRole === "Turing" && clientState.latestServerState.current_phase === "Turing_Action") ||
                                       (clientState.playerRole === "Scherbius" && clientState.latestServerState.current_phase === "Scherbius_Action");
            const isViewingCurrentPlayableRound = clientState.currentHistoryViewIndex === VIEWING_CURRENT_ROUND_INDEX();

            if (!isActivePlayerTurn || !isViewingCurrentPlayableRound || clientState.latestServerState.is_game_over) {
                e.preventDefault();
                return;
            }

            clientState.draggedCard = {
                id: e.target.dataset.cardId,
                value: e.target.dataset.cardValue === 'X' ? 'X' : parseInt(e.target.dataset.cardValue),
                originType: e.target.dataset.originType,
                originId: e.target.dataset.originId,
                originSlotIndex: e.target.dataset.originSlotIndex !== undefined ? parseInt(e.target.dataset.originSlotIndex) : null,
                element: e.target
            };
            e.dataTransfer.setData('text/plain', e.target.dataset.cardId);
            setTimeout(() => e.target.classList.add('dragging'), 0);
        });
        cardDiv.addEventListener('dragend', (e) => {
            e.target.classList.remove('dragging');
            clientState.draggedCard = { id: null, value: null, originType: null, originId: null, element: null, originSlotIndex: null };
        });
        return cardDiv;
    }

    function findCardInArrayById(cardArray, cardId) {
        return cardArray.find(card => card && card.id === cardId);
    }
    function removeCardFromArrayById(cardArray, cardId) {
        const index = cardArray.findIndex(card => card && card.id === cardId);
        if (index > -1) return cardArray.splice(index, 1)[0];
        return null;
    }

    function renderAllPlayableCardAreas() {
        const currentPlayerHandEl = getCurrentPlayerHandEl();
        if (!currentPlayerHandEl) {
            return;
        }

        let cardsEffectivelyInHandObjects = [...clientState.playerHandForTurn];
        Object.values(clientState.battlePlays).flat().forEach(playedCardObj => {
            if (playedCardObj) {
                cardsEffectivelyInHandObjects = cardsEffectivelyInHandObjects.filter(handCardObj => handCardObj.id !== playedCardObj.id);
            }
        });
        clientState.currentHandDisplayObjects = cardsEffectivelyInHandObjects;

        currentPlayerHandEl.innerHTML = '';
        const handOriginId = clientState.playerRole === 'Turing' ? 'turingHand' : 'scherbiusHand';
        clientState.currentHandDisplayObjects.forEach(cardObj => {
            // Pass playerRole to identify the card's owner for styling
            currentPlayerHandEl.appendChild(createCardElement(cardObj, 'hand', handOriginId, null, clientState.playerRole));
        });

        for (let i = 0; i < clientState.nBattles; i++) {
            const battleId = `battle_${i}`;
            const playerPlayedCardsAreaEl = rewardsDisplayEl.querySelector(`.player-played-cards-area[data-battle-id="${battleId}"]`);
            if (playerPlayedCardsAreaEl) {
                const slots = playerPlayedCardsAreaEl.querySelectorAll('.card-slot');
                slots.forEach(slot => slot.innerHTML = '');

                const cardsInThisBattle = clientState.battlePlays[battleId] || [];
                cardsInThisBattle.forEach((cardObj, slotIndex) => {
                    if (cardObj && slotIndex < slots.length) {
                        // Pass playerRole to identify the card's owner for styling
                        slots[slotIndex].appendChild(createCardElement(cardObj, 'battleSlot', battleId, slotIndex, clientState.playerRole));
                    }
                });
            }
        }
    }

    function handleCardDrop(targetBattleId, targetSlotIndex) {
        const { id: draggedCardId, originType, originId, originSlotIndex } = clientState.draggedCard;
        if (!draggedCardId) return;

        let cardObjectToMove = null;

        if (originType === 'hand') {
            cardObjectToMove = findCardInArrayById(clientState.playerHandForTurn, draggedCardId);
        } else if (originType === 'battleSlot') {
            if (clientState.battlePlays[originId] && clientState.battlePlays[originId][originSlotIndex]) {
                cardObjectToMove = clientState.battlePlays[originId][originSlotIndex];
                clientState.battlePlays[originId].splice(originSlotIndex, 1);
            }
        }

        if (!cardObjectToMove) {
            console.error("Dragged card object not found or couldn't be removed from origin!", clientState.draggedCard);
            renderAllPlayableCardAreas();
            return;
        }

        const targetBattleCards = clientState.battlePlays[targetBattleId];
        if (!targetBattleCards) {
            console.error("Target battle play array not found:", targetBattleId);
            if (originType === 'battleSlot') clientState.battlePlays[originId].splice(originSlotIndex, 0, cardObjectToMove);
            renderAllPlayableCardAreas();
            return;
        }

        const isMovingWithinSameBattle = originType === 'battleSlot' && originId === targetBattleId;
        if (targetBattleCards.length >= clientState.maxCardsPerBattle && !isMovingWithinSameBattle) {
            console.warn(`Battle ${targetBattleId} is full. Cannot add new card.`);
            if (originType === 'battleSlot') clientState.battlePlays[originId].splice(originSlotIndex, 0, cardObjectToMove);
            renderAllPlayableCardAreas();
            return;
        }

        targetBattleCards.splice(targetSlotIndex, 0, cardObjectToMove);
        clientState.battlePlays[targetBattleId] = targetBattleCards.filter(c => c !== null).slice(0, clientState.maxCardsPerBattle);

        renderAllPlayableCardAreas();
    }

    function handleCardDropToHand() {
        const { id: draggedCardId, originType, originId, originSlotIndex } = clientState.draggedCard;
        if (!draggedCardId || originType === 'hand') {
            renderAllPlayableCardAreas();
            return;
        }

        if (originType === 'battleSlot') {
            if (clientState.battlePlays[originId] && clientState.battlePlays[originId][originSlotIndex]) {
                clientState.battlePlays[originId].splice(originSlotIndex, 1);
            }
        }
        renderAllPlayableCardAreas();
    }

    function addDropListenersToElement(element, isHandArea, battleId = null, slotIndex = null) {
        element.addEventListener('dragover', e => {
            e.preventDefault();
            const isActivePlayerTurn = (clientState.playerRole === "Turing" && clientState.latestServerState.current_phase === "Turing_Action") ||
                                   (clientState.playerRole === "Scherbius" && clientState.latestServerState.current_phase === "Scherbius_Action");
            const isViewingCurrentPlayableRound = clientState.currentHistoryViewIndex === VIEWING_CURRENT_ROUND_INDEX();
            if (!isActivePlayerTurn || !isViewingCurrentPlayableRound || clientState.latestServerState.is_game_over) {
                return;
            }

            if (!isHandArea) {
                const targetBattleCards = clientState.battlePlays[battleId] || [];
                const cardInSlot = element.querySelector('.card');
                if (cardInSlot && cardInSlot !== clientState.draggedCard.element) {
                    element.classList.add('drag-over-full');
                } else if (targetBattleCards.length >= clientState.maxCardsPerBattle &&
                           !(clientState.draggedCard.originType === 'battleSlot' && clientState.draggedCard.originId === battleId) &&
                           !cardInSlot) {
                     element.classList.add('drag-over-full');
                } else {
                    element.classList.add('drag-over');
                }
            } else {
                 element.classList.add('drag-over');
            }
        });
        element.addEventListener('dragleave', e => {
            element.classList.remove('drag-over');
            element.classList.remove('drag-over-full');
        });
        element.addEventListener('drop', e => {
            e.preventDefault();
            element.classList.remove('drag-over');
            element.classList.remove('drag-over-full');

            const isActivePlayerTurn = (clientState.playerRole === "Turing" && clientState.latestServerState.current_phase === "Turing_Action") ||
                                   (clientState.playerRole === "Scherbius" && clientState.latestServerState.current_phase === "Scherbius_Action");
            const isViewingCurrentPlayableRound = clientState.currentHistoryViewIndex === VIEWING_CURRENT_ROUND_INDEX();
            if (!isActivePlayerTurn || !isViewingCurrentPlayableRound || clientState.latestServerState.is_game_over) {
                return;
            }

            if (isHandArea) {
                handleCardDropToHand();
            } else {
                const existingCardElement = element.querySelector('.card');
                if (existingCardElement && existingCardElement !== clientState.draggedCard.element) {
                    console.warn("Slot is already occupied by a different card.");
                    renderAllPlayableCardAreas();
                    return;
                }
                handleCardDrop(battleId, slotIndex);
            }
        });
    }
    addDropListenersToElement(turingHandEl, true);
    addDropListenersToElement(scherbiusHandEl, true);


    // --- Main Battle Display (Unified for Current and Historical) ---
    function renderMainBattleDisplay(currentServerState) {
        const historyIdx = clientState.currentHistoryViewIndex;
        const isViewingCurrentRoundSetup = (historyIdx === VIEWING_CURRENT_ROUND_INDEX() && !currentServerState.is_game_over);
        
        rewardsDisplayEl.innerHTML = ''; 

        if (isViewingCurrentRoundSetup) {
            const playerRole = clientState.playerRole;
            battleZoneTitleEl.textContent = `Your Turn (${playerRole}) - Drag Cards to Battles`;
            
            const numBattlesToDisplay = currentServerState.n_battles;
            const maxCardsPerBattleForDisplay = clientState.maxCardsPerBattle;

            const currentRewards = currentServerState.rewards;
            const opponentObservedPlays = currentServerState.opponent_observed_plays; 

            for (let i = 0; i < numBattlesToDisplay; i++) {
                const battleId = `battle_${i}`;
                const battleDiv = document.createElement('div');
                battleDiv.classList.add('battle-item');

                let rewardInfoHTML = `<div class="reward-info"><h4>Battle ${i + 1} (Current)</h4>`;
                const vpReward = currentRewards.vp_rewards[i];
                const cardRewardArray = currentRewards.card_rewards[i];
                let hasRewards = false;
                let rewardsContent = '<div class="display-card-container">';
                if (vpReward > 0) {
                    rewardsContent += `<div class="display-card vp-reward">${vpReward}</div>`;
                    hasRewards = true;
                }
                if (cardRewardArray && cardRewardArray.length > 0) {
                    cardRewardArray.forEach(cardVal => {
                        rewardsContent += `<div class="display-card">${cardVal}</div>`;
                    });
                    hasRewards = true;
                }
                rewardsContent += '</div>';
                if (hasRewards) {
                    rewardInfoHTML += `<p>Potential Rewards: ${rewardsContent}</p>`;
                } else {
                    rewardInfoHTML += `<p>Potential Rewards: <div class="display-card-container"></div></p>`;
                }
                rewardInfoHTML += `</div>`;

                let opponentInfoHTML = `<div class="opponent-observed-info">`;
                const opponentPlayForBattle = (opponentObservedPlays && opponentObservedPlays[i]) ? opponentObservedPlays[i] : [];
                let opponentCardsContent = '<div class="display-card-container">';
                const opponentName = playerRole === 'Turing' ? 'Scherbius' : 'Turing';

                if (opponentPlayForBattle.length > 0) {
                    opponentPlayForBattle.forEach(cardVal => {
                        const cardEl = document.createElement('div');
                        cardEl.classList.add('display-card');
                        if (opponentName === 'Scherbius') {
                            cardEl.classList.add('scherbius-opponent-card');
                        } else if (opponentName === 'Turing') {
                            cardEl.classList.add('turing-opponent-card');
                        }
                        cardEl.textContent = cardVal;
                        opponentCardsContent += cardEl.outerHTML;
                    });
                }
                opponentCardsContent += '</div>';
                
                let opponentStatusText = `${opponentName} Will Play: ${opponentCardsContent}`;
                if (playerRole === 'Turing' && currentServerState.scherbius_did_encrypt) {
                    opponentStatusText += ` (Encrypted!)`;
                } else if (playerRole === 'Scherbius') {
                    opponentStatusText = `${opponentName}'s plays are hidden until round resolution.`;
                }
                opponentInfoHTML += `<p>${opponentStatusText}</p></div>`;

                const playerPlayedCardsArea = document.createElement('div');
                playerPlayedCardsArea.classList.add('player-played-cards-area');
                playerPlayedCardsArea.dataset.battleId = battleId;
                playerPlayedCardsArea.innerHTML = `<h4>Your Plays for Battle ${i+1}:</h4>`;
                const slotsContainer = document.createElement('div');
                slotsContainer.classList.add('card-slots-container');
                
                for (let slotIdx = 0; slotIdx < maxCardsPerBattleForDisplay; slotIdx++) {
                    const slotDiv = document.createElement('div');
                    slotDiv.classList.add('card-slot');
                    slotDiv.dataset.battleId = battleId;
                    slotDiv.dataset.slotIndex = slotIdx;
                    addDropListenersToElement(slotDiv, false, battleId, slotIdx);
                    slotsContainer.appendChild(slotDiv);
                }
                playerPlayedCardsArea.appendChild(slotsContainer);
                
                battleDiv.innerHTML = rewardInfoHTML + opponentInfoHTML;
                battleDiv.appendChild(playerPlayedCardsArea);
                rewardsDisplayEl.appendChild(battleDiv);
            }
            renderAllPlayableCardAreas();

        } else { 
            const roundData = clientState.roundHistory[historyIdx];
            if (!roundData || !roundData.battles) {
                if (currentServerState.is_game_over && historyIdx === VIEWING_CURRENT_ROUND_INDEX() && clientState.roundHistory.length > 0) {
                     battleZoneTitleEl.textContent = "Game Over - Final Round Details";
                     const lastRoundData = clientState.roundHistory[clientState.roundHistory.length -1];
                     if (lastRoundData && lastRoundData.battles) {
                        // Fallthrough to historical rendering with lastRoundData
                     } else {
                        console.error("Historical round data or battles missing for index:", historyIdx);
                        battleZoneTitleEl.textContent = "Error displaying historical round.";
                        return;
                     }
                } else {
                    console.error("Historical round data or battles missing for index:", historyIdx, "State:", clientState);
                    battleZoneTitleEl.textContent = "Error displaying historical round.";
                    return;
                }
            }
            const displayData = (currentServerState.is_game_over && historyIdx === VIEWING_CURRENT_ROUND_INDEX() && clientState.roundHistory.length > 0) ? clientState.roundHistory[clientState.roundHistory.length -1] : roundData;

            if (!displayData || !displayData.battles) { 
                battleZoneTitleEl.textContent = "No round data to display.";
                return;
            }

            battleZoneTitleEl.textContent = `Details for Past Round ${displayData.round_number}`;

            (displayData.battles).forEach(battle => {
                const battleDiv = document.createElement('div');
                battleDiv.classList.add('battle-item', 'historical-battle-item');

                let titleHTML = `<h4>Battle ${battle.id + 1} (Round ${displayData.round_number})</h4>`;
                let battleOutcomeHTML = '';
                if (battle.winner) {
                    let outcomeClass = '';
                    let outcomeText = '';
                    if (battle.winner === 'Turing') { outcomeClass = 'turing-wins'; outcomeText = 'Turing won this battle.'; }
                    else if (battle.winner === 'Scherbius') { outcomeClass = 'scherbius-wins'; outcomeText = 'Scherbius won this battle.'; }
                    else if (battle.winner === 'Draw') { outcomeClass = 'draw'; outcomeText = 'Battle was a draw.';}
                    if (outcomeText) battleOutcomeHTML = `<p class="battle-outcome ${outcomeClass}">${outcomeText}</p>`;
                }
                
                let rewardInfoHTML = `<div class="reward-info">`;
                let rewardsContent = '<div class="display-card-container">';
                if (battle.rewards_available_to_turing.vp > 0) rewardsContent += `<div class="display-card vp-reward">${battle.rewards_available_to_turing.vp}</div>`;
                if (battle.rewards_available_to_turing.cards && battle.rewards_available_to_turing.cards.length > 0) {
                    battle.rewards_available_to_turing.cards.forEach(cardVal => { rewardsContent += `<div class="display-card">${cardVal}</div>`; });
                }
                rewardsContent += '</div>';
                rewardInfoHTML += `<p>Rewards Available: ${rewardsContent}</p></div>`;

                // Scherbius committed cards area
                let scherbiusInfoHTML = `<div class="scherbius-observed-info">`; // Re-using class for consistency, styled differently for historical
                let scherbiusCardsContent = '<div class="display-card-container">';
                if (battle.scherbius_committed_cards.length > 0) {
                    battle.scherbius_committed_cards.forEach(cardVal => {
                        // Use .scherbius-card for historical Scherbius cards
                        scherbiusCardsContent += `<div class="display-card scherbius-card">${cardVal}</div>`;
                    });
                } else {
                     scherbiusCardsContent += `<span>(No cards)</span>`;
                }
                scherbiusCardsContent += '</div>';
                let scherbiusEncryptedText = displayData.scherbius_encrypted_this_round ? " (Encrypted)" : "";
                scherbiusInfoHTML += `<h4>Scherbius Committed${scherbiusEncryptedText}:</h4>${scherbiusCardsContent}</div>`;
                
                // Turing played cards area
                let turingPlayedHTML = `<div class="static-played-cards-area"><h4>Turing Played:</h4>`;
                let turingCardsContent = '<div class="display-card-container">';
                if (battle.turing_played_cards.length > 0) {
                    battle.turing_played_cards.forEach(cardVal => {
                         // Use .turing-summary-card for historical Turing cards
                         turingCardsContent += `<div class="display-card turing-summary-card">${cardVal}</div>`;
                    });
                } else {
                    turingCardsContent += `<span>(No cards)</span>`;
                }
                turingCardsContent += '</div>';
                turingPlayedHTML += `${turingCardsContent}</div>`;

                battleDiv.innerHTML = titleHTML + battleOutcomeHTML + rewardInfoHTML + scherbiusInfoHTML + turingPlayedHTML;
                rewardsDisplayEl.appendChild(battleDiv);
            });
        }
    }
    
    // --- Event Listeners ---
    playAsTuringBtn.addEventListener('click', () => {
        gameOverMessageEl.style.display = 'none';
        fetchApi('/new_game', 'POST', { player_role: "Turing" });
    });
    playAsScherbiusBtn.addEventListener('click', () => {
        gameOverMessageEl.style.display = 'none';
        fetchApi('/new_game', 'POST', { player_role: "Scherbius" });
    });

    function handleSubmitAction() {
        const finalPlayerStrategyValues = [];
        for (let i = 0; i < clientState.nBattles; i++) {
            const battleId = `battle_${i}`;
            const cardsInBattle = clientState.battlePlays[battleId] || [];
            finalPlayerStrategyValues.push(cardsInBattle.map(cardObj => {
                return cardObj.value === 'X' ? 'X' : parseInt(cardObj.value);
            }));
        }
        
        const payload = { 
            player_strategy: finalPlayerStrategyValues,
        };

        if (clientState.playerRole === 'Scherbius') {
            payload.scherbius_encrypts = scherbiusEncryptCheckbox.checked;
        }
        
        console.log("Submitting player action to backend:", payload);
        fetchApi('/submit_player_action', 'POST', payload);
    }

    turingSubmitActionBtn.addEventListener('click', handleSubmitAction);
    scherbiusSubmitActionBtn.addEventListener('click', handleSubmitAction);


    // --- UI State Management for Current vs. Historical View ---
    function manageRoundViewSpecificUI() {
        const serverState = clientState.latestServerState;
        if (!serverState) {
            turingControlsEl.style.display = 'none';
            scherbiusControlsEl.style.display = 'none';
            gameAreaEl.style.display = 'none'; 
            historicalRoundViewAreaEl.style.display = 'none';
            return;
        }

        const isViewingCurrent = clientState.currentHistoryViewIndex === VIEWING_CURRENT_ROUND_INDEX();
        const isGameOver = serverState.is_game_over;
        const playerRole = clientState.playerRole;
        
        const isTuringTurnNow = !isGameOver && isViewingCurrent && playerRole === "Turing" && serverState.current_phase === "Turing_Action";
        const isScherbiusTurnNow = !isGameOver && isViewingCurrent && playerRole === "Scherbius" && serverState.current_phase === "Scherbius_Action";

        turingControlsEl.style.display = isTuringTurnNow ? 'block' : 'none';
        if (isTuringTurnNow) { 
            turingHandAreaEl.style.display = 'block';
            turingHandEl.style.display = 'flex'; 
            turingSubmitActionBtn.style.display = 'block';
            turingSubmitActionBtn.disabled = false;
        }

        scherbiusControlsEl.style.display = isScherbiusTurnNow ? 'block' : 'none';
        if (isScherbiusTurnNow) { 
            scherbiusHandAreaEl.style.display = 'block';
            scherbiusHandEl.style.display = 'flex'; 
            scherbiusSubmitActionBtn.style.display = 'block';
            scherbiusSubmitActionBtn.disabled = false;
        }
        
        gameAreaEl.style.display = 'block';
        historicalRoundViewAreaEl.style.display = 'block';
    }

    // --- Navigation Controls Update ---
    function updateHistoryNavigationControls() {
        const historyLen = clientState.roundHistory.length;
        const viewingCurrent = clientState.currentHistoryViewIndex === VIEWING_CURRENT_ROUND_INDEX();

        prevRoundBtnEl.disabled = clientState.currentHistoryViewIndex <= 0;
        nextRoundBtnEl.disabled = viewingCurrent;

        if (viewingCurrent) {
            if (clientState.latestServerState && clientState.latestServerState.is_game_over) {
                 historicalRoundIndicatorEl.textContent = `Game Over. Viewing last round actions.`;
            } else if (historyLen > 0) {
                historicalRoundIndicatorEl.textContent = `Viewing Current Round (Prepare for Round ${historyLen + 1})`;
            } else {
                historicalRoundIndicatorEl.textContent = "Viewing Current Round (Prepare for Round 1)";
            }
        } else {
            if (historyLen > 0 && clientState.currentHistoryViewIndex < historyLen) {
                historicalRoundIndicatorEl.textContent = `Viewing Past Round ${clientState.roundHistory[clientState.currentHistoryViewIndex].round_number} of ${historyLen}`;
            } else { 
                historicalRoundIndicatorEl.textContent = "History Navigation"; 
            }
        }
    }
    
    prevRoundBtnEl.addEventListener('click', () => {
        if (clientState.currentHistoryViewIndex > 0) {
            clientState.currentHistoryViewIndex--;
            renderMainBattleDisplay(clientState.latestServerState);
            updateHistoryNavigationControls();
            manageRoundViewSpecificUI();
        }
    });

    nextRoundBtnEl.addEventListener('click', () => {
        if (clientState.currentHistoryViewIndex < VIEWING_CURRENT_ROUND_INDEX()) {
            clientState.currentHistoryViewIndex++;
            
            const isNowViewingCurrentPlayable = clientState.currentHistoryViewIndex === VIEWING_CURRENT_ROUND_INDEX();
            const serverState = clientState.latestServerState;

            if (isNowViewingCurrentPlayable && serverState && !serverState.is_game_over) {
                 const isPlayerActionPhase = (clientState.playerRole === "Turing" && serverState.current_phase === "Turing_Action") ||
                                    (clientState.playerRole === "Scherbius" && serverState.current_phase === "Scherbius_Action");
                if (isPlayerActionPhase) {
                    clientState.playerHandForTurn = [...(serverState.player_hand || [])];
                    clientState.battlePlays = {};
                    for (let i = 0; i < clientState.nBattles; i++) {
                        clientState.battlePlays[`battle_${i}`] = [];
                    }
                     if (clientState.playerRole === 'Scherbius') {
                        scherbiusEncryptCheckbox.checked = false;
                    }
                }
            }
            renderMainBattleDisplay(serverState);
            updateHistoryNavigationControls();
            manageRoundViewSpecificUI();
        }
    });

    // --- Initial Setup ---
    gameAreaEl.style.display = 'none';
    gameOverMessageEl.style.display = 'none';
    document.querySelector('.game-controls').style.display = 'block';
    
    clientState.currentHistoryViewIndex = VIEWING_CURRENT_ROUND_INDEX();
    updateHistoryNavigationControls();
    manageRoundViewSpecificUI(); 
});