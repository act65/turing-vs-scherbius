// script.js
document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const turingScoreEl = document.getElementById('turingScore');
    const scherbiusScoreEl = document.getElementById('scherbiusScore');
    const maxVictoryPointsEl = document.getElementById('maxVictoryPoints');
    const winnerEl = document.getElementById('winner');
    const newGameBtn = document.getElementById('newGameBtn');
    const submitTuringActionBtn = document.getElementById('submitTuringActionBtn');
    const gameAreaEl = document.getElementById('gameArea');
    const gameOverMessageEl = document.getElementById('gameOverMessage');
    const rewardsDisplayEl = document.getElementById('rewardsDisplay');
    const turingHandEl = document.getElementById('turingHand');
    const battleZoneTitleEl = document.getElementById('battleZoneTitle');

    // --- DOM Elements for Historical View ---
    const historicalRoundViewAreaEl = document.getElementById('historicalRoundViewArea');
    const prevRoundBtnEl = document.getElementById('prevRoundBtn');
    const nextRoundBtnEl = document.getElementById('nextRoundBtn');
    const historicalRoundIndicatorEl = document.getElementById('historicalRoundIndicator');
    // const historicalRoundContentEl = document.getElementById('historicalRoundContent'); // Kept for structure, but content not primary display

    // --- Client-Side Game State ---
    let clientState = {
        initialHandForTurn: [], currentHandDisplayObjects: [],
        battlePlays: {},
        nBattles: 0,
        maxCardsPerBattle: 0,
        draggedCard: { id: null, value: null, originType: null, originId: null, element: null, originSlotIndex: null },
        roundHistory: [],
        currentHistoryViewIndex: 0,
        latestServerState: null
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
                throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
            }
            const state = await response.json();

            updateGlobalUI(state, endpoint); // Pass endpoint to UI updater
        } catch (error) {
            console.error(`API Error (${endpoint}):`, error);
            alert(`Error: ${error.message}`);
        }
    }

    // --- Main UI Update Function ---
    function updateGlobalUI(serverState, endpointCalled) {
        console.log("Server state received:", serverState, "from endpoint:", endpointCalled);
        gameAreaEl.style.display = 'block';

        turingScoreEl.textContent = serverState.turing_points;
        scherbiusScoreEl.textContent = "???";
        maxVictoryPointsEl.textContent = serverState.max_victory_points;
        
        clientState.nBattles = serverState.n_battles;
        clientState.maxCardsPerBattle = serverState.max_cards_per_battle;
        clientState.roundHistory = serverState.round_history || [];
        clientState.latestServerState = serverState; // Store the latest complete server state

        const newRoundHistoryLength = clientState.roundHistory.length;

        // Determine currentHistoryViewIndex based on game state and the action performed
        if (serverState.is_game_over) {
            // If game is over, view the last completed round
            if (newRoundHistoryLength > 0) {
                clientState.currentHistoryViewIndex = newRoundHistoryLength - 1;
            } else {
                clientState.currentHistoryViewIndex = VIEWING_CURRENT_ROUND_INDEX(); // Should be 0
            }
        } else { // Game not over
            if (endpointCalled === '/submit_turing_action' && newRoundHistoryLength > 0) {
                // After submitting an action, view the results of the round that was just completed
                clientState.currentHistoryViewIndex = newRoundHistoryLength - 1;
            } else {
                // For new_game, initial load, or other state updates, view the current action phase setup
                clientState.currentHistoryViewIndex = VIEWING_CURRENT_ROUND_INDEX();
            }
        }
        
        // Reset hand and battle plays only if viewing the current round's setup phase
        if (serverState.current_phase === "Turing_Action" && clientState.currentHistoryViewIndex === VIEWING_CURRENT_ROUND_INDEX()) {
            clientState.initialHandForTurn = [...(serverState.turing_hand || [])]; 
            clientState.battlePlays = {};
            for(let i=0; i < clientState.nBattles; i++) clientState.battlePlays[`battle_${i}`] = [];
        }
        
        renderMainBattleDisplay(serverState);
        
        historicalRoundViewAreaEl.style.display = 'block';

        updateHistoryNavigationControls(); 
        manageRoundViewSpecificUI();

        if (serverState.is_game_over) {
            gameOverMessageEl.style.display = 'block';
            winnerEl.textContent = serverState.winner;
            // submitTuringActionBtn.disabled = true; // Handled by manageRoundViewSpecificUI
        } else {
            gameOverMessageEl.style.display = 'none';
        }
    }

    // --- Card Rendering and D&D ---
    function createCardElement(cardObject, originType, originId, originSlotIndex = null) {
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

        cardDiv.addEventListener('dragstart', (e) => {
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
        let cardsEffectivelyInHandObjects = [...clientState.initialHandForTurn];
        Object.values(clientState.battlePlays).flat().forEach(playedCardObj => {
            if (playedCardObj) {
                cardsEffectivelyInHandObjects = cardsEffectivelyInHandObjects.filter(handCardObj => handCardObj.id !== playedCardObj.id);
            }
        });
        clientState.currentHandDisplayObjects = cardsEffectivelyInHandObjects;

        turingHandEl.innerHTML = '';
        clientState.currentHandDisplayObjects.forEach(cardObj => {
            turingHandEl.appendChild(createCardElement(cardObj, 'hand', 'turingHand'));
        });

        for (let i = 0; i < clientState.nBattles; i++) {
            const battleId = `battle_${i}`;
            const battleAreaEl = rewardsDisplayEl.querySelector(`.turing-played-cards-area[data-battle-id="${battleId}"]`);
            if (battleAreaEl) {
                const slots = battleAreaEl.querySelectorAll('.card-slot');
                slots.forEach(slot => slot.innerHTML = ''); 

                const cardsInThisBattle = clientState.battlePlays[battleId] || [];
                cardsInThisBattle.forEach((cardObj, slotIndex) => {
                    if (cardObj && slotIndex < slots.length) {
                        slots[slotIndex].appendChild(createCardElement(cardObj, 'battleSlot', battleId, slotIndex));
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
            cardObjectToMove = findCardInArrayById(clientState.initialHandForTurn, draggedCardId);
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
        
        if (targetBattleCards.length >= clientState.maxCardsPerBattle && 
            !(originType === 'battleSlot' && originId === targetBattleId)) {
            console.warn(`Battle ${targetBattleId} is full. Cannot add new card.`);
            if (originType === 'battleSlot') clientState.battlePlays[originId].splice(originSlotIndex, 0, cardObjectToMove);
            renderAllPlayableCardAreas();
            return;
        }
        
        if (originType === 'battleSlot' && originId === targetBattleId) {
            targetBattleCards.splice(targetSlotIndex, 0, cardObjectToMove);
        } else {
            if (targetBattleCards.length < clientState.maxCardsPerBattle) {
                targetBattleCards.splice(targetSlotIndex, 0, cardObjectToMove);
            } else {
                console.warn("Target battle is full, cannot place card (safeguard).");
                if (originType === 'battleSlot') clientState.battlePlays[originId].splice(originSlotIndex, 0, cardObjectToMove);
            }
        }
        
        clientState.battlePlays[targetBattleId] = clientState.battlePlays[targetBattleId].filter(c => c !== null);
        renderAllPlayableCardAreas();
    }
    
    function handleCardDropToHand() {
        const { id: draggedCardId, originType, originId, originSlotIndex } = clientState.draggedCard;
        if (!draggedCardId || originType === 'hand') {
            renderAllPlayableCardAreas();
            return;
        }

        // let cardObjectToMove = null; // Not strictly needed if we just re-render
        if (originType === 'battleSlot') {
            if (clientState.battlePlays[originId] && clientState.battlePlays[originId][originSlotIndex]) {
                /* cardObjectToMove = */ clientState.battlePlays[originId].splice(originSlotIndex, 1)[0];
            }
        }
        // if (!cardObjectToMove) { // Card might have been moved by other means or error
        //     console.error("Card to return to hand not found in origin slot.");
        // }
        renderAllPlayableCardAreas();
    }

    function addDropListenersToElement(element, isHandArea, battleId = null, slotIndex = null) {
        element.addEventListener('dragover', e => { 
            e.preventDefault(); 
            if (!isHandArea) {
                const targetBattleCards = clientState.battlePlays[battleId] || [];
                const cardInSlot = element.querySelector('.card');
                if (cardInSlot && !(clientState.draggedCard.originType === 'battleSlot' && clientState.draggedCard.originId === battleId)) {
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
            if (isHandArea) {
                handleCardDropToHand();
            } else {
                const existingCardElement = element.querySelector('.card');
                if (existingCardElement && existingCardElement !== clientState.draggedCard.element) {
                    console.warn("Slot is already occupied by a different card.");
                    renderAllPlayableCardAreas(); // Re-render to reflect actual state
                    return;
                }
                handleCardDrop(battleId, slotIndex);
            }
        });
    }
    addDropListenersToElement(turingHandEl, true);


    // --- Main Battle Display (Unified for Current and Historical) ---
    function renderMainBattleDisplay(serverOrFullStateData) { // serverOrFullStateData is latestServerState
        const historyIdx = clientState.currentHistoryViewIndex;
        const isViewingCurrentRound = (historyIdx === VIEWING_CURRENT_ROUND_INDEX());
        
        rewardsDisplayEl.innerHTML = ''; 

        if (isViewingCurrentRound) {
            battleZoneTitleEl.textContent = "Current Battles & Rewards (Drag Your Cards Here)";
            const numBattlesToDisplay = serverOrFullStateData.n_battles;
            const maxCardsPerBattleForDisplay = clientState.maxCardsPerBattle; // Use current game setting for slots

            const currentRewards = serverOrFullStateData.rewards;
            const currentScherbiusObserved = serverOrFullStateData.scherbius_observed_plays;

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

                let scherbiusInfoHTML = `<div class="scherbius-observed-info">`;
                const scherbiusPlayForBattle = (currentScherbiusObserved && currentScherbiusObserved[i]) ? currentScherbiusObserved[i] : [];
                let scherbiusCardsContent = '<div class="display-card-container">';
                if (scherbiusPlayForBattle.length > 0) {
                    scherbiusPlayForBattle.forEach(cardVal => {
                        const cardEl = document.createElement('div');
                        cardEl.classList.add('display-card', 'scherbius-card');
                        cardEl.textContent = cardVal;
                        scherbiusCardsContent += cardEl.outerHTML;
                    });
                }
                scherbiusCardsContent += '</div>';
                scherbiusInfoHTML += `<p>Scherbius Will Play: ${scherbiusCardsContent}</p></div>`;

                const turingPlayedCardsArea = document.createElement('div');
                turingPlayedCardsArea.classList.add('turing-played-cards-area');
                turingPlayedCardsArea.dataset.battleId = battleId;
                for (let slotIdx = 0; slotIdx < maxCardsPerBattleForDisplay; slotIdx++) {
                    const slotDiv = document.createElement('div');
                    slotDiv.classList.add('card-slot');
                    slotDiv.dataset.battleId = battleId;
                    slotDiv.dataset.slotIndex = slotIdx;
                    addDropListenersToElement(slotDiv, false, battleId, slotIdx);
                    turingPlayedCardsArea.appendChild(slotDiv);
                }
                
                battleDiv.innerHTML = rewardInfoHTML + scherbiusInfoHTML;
                battleDiv.appendChild(turingPlayedCardsArea);
                rewardsDisplayEl.appendChild(battleDiv);
            }
            renderAllPlayableCardAreas();

        } else { // RENDER HISTORICAL ROUND (STATIC)
            const roundData = clientState.roundHistory[historyIdx];
            if (!roundData || !roundData.battles) {
                console.error("Historical round data or battles missing for index:", historyIdx);
                battleZoneTitleEl.textContent = "Error displaying historical round.";
                return;
            }
            battleZoneTitleEl.textContent = `Details for Past Round ${roundData.round_number}`;

            roundData.battles.forEach(battle => {
                const battleDiv = document.createElement('div');
                battleDiv.classList.add('battle-item', 'historical-battle-item');

                let titleHTML = `<h4>Battle ${battle.id + 1} (Round ${roundData.round_number})</h4>`;

                let battleOutcomeHTML = '';
                if (battle.winner) {
                    let outcomeClass = '';
                    let outcomeText = '';
                    if (battle.winner === 'Turing') {
                        outcomeClass = 'turing-wins';
                        outcomeText = 'Turing won this battle.';
                    } else if (battle.winner === 'Scherbius') {
                        outcomeClass = 'scherbius-wins';
                        outcomeText = 'Scherbius won this battle.';
                    } else if (battle.winner === 'Draw') {
                        outcomeClass = 'draw';
                        outcomeText = 'Battle was a draw.';
                    }
                    if (outcomeText) {
                        battleOutcomeHTML = `<p class="battle-outcome ${outcomeClass}">${outcomeText}</p>`;
                    }
                }
                
                let rewardInfoHTML = `<div class="reward-info">`;
                let rewardsContent = '<div class="display-card-container">';
                let hasRewards = false;
                if (battle.rewards_available_to_turing.vp > 0) {
                    rewardsContent += `<div class="display-card vp-reward">${battle.rewards_available_to_turing.vp}</div>`;
                    hasRewards = true;
                }
                if (battle.rewards_available_to_turing.cards && battle.rewards_available_to_turing.cards.length > 0) {
                    battle.rewards_available_to_turing.cards.forEach(cardVal => {
                        rewardsContent += `<div class="display-card">${cardVal}</div>`;
                    });
                    hasRewards = true;
                }
                rewardsContent += '</div>';
                rewardInfoHTML += `<p>Rewards Available: ${rewardsContent}</p></div>`;

                let scherbiusInfoHTML = `<div class="scherbius-observed-info">`;
                let scherbiusCardsContent = '<div class="display-card-container">';
                if (battle.scherbius_committed_cards.length > 0) {
                    battle.scherbius_committed_cards.forEach(cardVal => {
                        const cardEl = document.createElement('div');
                        cardEl.classList.add('display-card', 'scherbius-card');
                        cardEl.textContent = cardVal;
                        scherbiusCardsContent += cardEl.outerHTML;
                    });
                }
                scherbiusCardsContent += '</div>';
                scherbiusInfoHTML += `<p>Scherbius Committed: ${scherbiusCardsContent}</p></div>`;
                
                let turingPlayedHTML = `<div class="turing-played-cards-area static-played-cards-area"><h4>Turing Played:</h4>`;
                let turingCardsContent = '<div class="display-card-container">';
                if (battle.turing_played_cards.length > 0) {
                    battle.turing_played_cards.forEach(cardVal => {
                         turingCardsContent += `<div class="display-card turing-summary-card">${cardVal}</div>`;
                    });
                }
                turingCardsContent += '</div>';
                turingPlayedHTML += `${turingCardsContent}</div>`;

                battleDiv.innerHTML = titleHTML + battleOutcomeHTML + rewardInfoHTML + scherbiusInfoHTML + turingPlayedHTML;
                rewardsDisplayEl.appendChild(battleDiv);
            });
        }
    }
    
    // --- Event Listeners ---
    newGameBtn.addEventListener('click', () => {
        gameOverMessageEl.style.display = 'none';
        // clientState.roundHistory = []; // Will be reset by server response via updateGlobalUI
        fetchApi('/new_game', 'POST');
    });

    submitTuringActionBtn.addEventListener('click', () => {
        const finalTuringStrategyValues = [];
        for (let i = 0; i < clientState.nBattles; i++) {
            const battleId = `battle_${i}`;
            const cardsInBattle = clientState.battlePlays[battleId] || [];
            finalTuringStrategyValues.push(cardsInBattle.map(cardObj => {
                return cardObj.value === 'X' ? 'X' : parseInt(cardObj.value);
            }));
        }
        
        const payload = { 
            turing_strategy: finalTuringStrategyValues,
        };
        console.log("Submitting to backend (values):", payload);
        fetchApi('/submit_turing_action', 'POST', payload);
    });

    // --- Initial Setup ---
    gameAreaEl.style.display = 'none';
    gameOverMessageEl.style.display = 'none';
    // historicalRoundContentEl.style.display = 'none'; 
    clientState.currentHistoryViewIndex = VIEWING_CURRENT_ROUND_INDEX(); // Initially 0
    updateHistoryNavigationControls();
    manageRoundViewSpecificUI(); // Will hide controls as latestServerState is null

    // --- UI State Management for Current vs. Historical View ---
    function manageRoundViewSpecificUI() {
        const isViewingCurrent = clientState.currentHistoryViewIndex === VIEWING_CURRENT_ROUND_INDEX();
        // Default to true for isGameOver if latestServerState is null, to be safe (e.g. disable buttons initially)
        const isGameOver = clientState.latestServerState ? clientState.latestServerState.is_game_over : true; 
        const isTuringActionPhase = clientState.latestServerState ? clientState.latestServerState.current_phase === "Turing_Action" : false;

        // Show Turing's hand and action controls only if:
        // 1. Viewing the current round's setup (not a past round summary)
        // 2. It's actually Turing's action phase
        // 3. The game is not over
        const showTuringControls = isViewingCurrent && isTuringActionPhase && !isGameOver;

        const turingHandArea = document.getElementById('turingHandArea');
        const turingActionControlsTitle = document.getElementById('turingActionControls').querySelector('h3');

        if (turingHandArea) turingHandArea.style.display = showTuringControls ? 'block' : 'none';
        if (turingActionControlsTitle) turingActionControlsTitle.style.display = showTuringControls ? 'block' : 'none';
        turingHandEl.style.display = showTuringControls ? 'flex' : 'none';
        
        submitTuringActionBtn.style.display = showTuringControls ? 'block' : 'none';
        submitTuringActionBtn.disabled = !showTuringControls;
        
        gameAreaEl.style.display = clientState.latestServerState ? 'block' : 'none'; 
        historicalRoundViewAreaEl.style.display = clientState.latestServerState ? 'block' : 'none';
    }

    // --- Navigation Controls Update ---
    function updateHistoryNavigationControls() {
        const historyLen = clientState.roundHistory.length;
        const viewingCurrent = clientState.currentHistoryViewIndex === VIEWING_CURRENT_ROUND_INDEX();

        prevRoundBtnEl.disabled = clientState.currentHistoryViewIndex <= 0;
        nextRoundBtnEl.disabled = viewingCurrent;

        if (viewingCurrent) {
            if (historyLen > 0) {
                historicalRoundIndicatorEl.textContent = `Viewing Current Round (Prepare for Round ${historyLen + 1})`;
            } else {
                historicalRoundIndicatorEl.textContent = "Viewing Current Round (Prepare for Round 1)";
            }
        } else {
            if (historyLen > 0 && clientState.currentHistoryViewIndex < historyLen) {
                historicalRoundIndicatorEl.textContent = `Viewing Past Round ${clientState.roundHistory[clientState.currentHistoryViewIndex].round_number} of ${historyLen}`;
            } else {
                // Should not happen if navigation is correct, but as a fallback:
                historicalRoundIndicatorEl.textContent = "History Navigation"; 
            }
        }
    }

    prevRoundBtnEl.addEventListener('click', () => {
        if (clientState.currentHistoryViewIndex > 0) {
            clientState.currentHistoryViewIndex--;
            renderMainBattleDisplay(clientState.latestServerState); // Re-render with existing server state
            updateHistoryNavigationControls();
            manageRoundViewSpecificUI();
        }
    });

    nextRoundBtnEl.addEventListener('click', () => {
        if (clientState.currentHistoryViewIndex < VIEWING_CURRENT_ROUND_INDEX()) {
            clientState.currentHistoryViewIndex++;

            // **BUG FIX STARTS HERE**
            // If we are now viewing the current round's action phase,
            // ensure the hand and battle plays are reset for the new turn.
            if (clientState.currentHistoryViewIndex === VIEWING_CURRENT_ROUND_INDEX() &&
                clientState.latestServerState &&
                clientState.latestServerState.current_phase === "Turing_Action") {
                
                // Re-initialize hand from the latest server state for the new turn
                clientState.initialHandForTurn = [...(clientState.latestServerState.turing_hand || [])];
                
                // nBattles and maxCardsPerBattle should already be up-to-date in clientState
                // from the last updateGlobalUI call, reflecting the current round's parameters.
                
                // Reset battlePlays for the new turn
                clientState.battlePlays = {};
                for (let i = 0; i < clientState.nBattles; i++) { // clientState.nBattles is for the current round
                    clientState.battlePlays[`battle_${i}`] = [];
                }
            }
            // **BUG FIX ENDS HERE**

            renderMainBattleDisplay(clientState.latestServerState);
            updateHistoryNavigationControls();
            manageRoundViewSpecificUI();
        }
    });
});