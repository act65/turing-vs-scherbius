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
    // REMOVED: const lastRoundSummaryAreaEl = document.getElementById('lastRoundSummaryArea');
    // REMOVED: const lastRoundInfoEl = document.getElementById('lastRoundInfo');
    // REMOVED: const lastRoundSummaryBattlesEl = document.getElementById('lastRoundSummaryBattles');
    const rewardsDisplayEl = document.getElementById('rewardsDisplay');
    const turingHandEl = document.getElementById('turingHand');
    const battleZoneTitleEl = document.getElementById('battleZoneTitle'); // Added

    // --- DOM Elements for Historical View ---
    const historicalRoundViewAreaEl = document.getElementById('historicalRoundViewArea');
    const prevRoundBtnEl = document.getElementById('prevRoundBtn');
    const nextRoundBtnEl = document.getElementById('nextRoundBtn');
    const historicalRoundIndicatorEl = document.getElementById('historicalRoundIndicator');
    // const historicalRoundContentEl = document.getElementById('historicalRoundContent'); // Kept for structure, but content not primary display
    // const historicalRoundInfoEl = document.getElementById('historicalRoundInfo'); // Not used for battle display
    // const historicalRoundBattlesEl = document.getElementById('historicalRoundBattles'); // Not used for battle display

    // --- Client-Side Game State ---
    let clientState = {
        initialHandForTurn: [], currentHandDisplayObjects: [],
        battlePlays: {},
        nBattles: 0,
        maxCardsPerBattle: 0,
        draggedCard: { id: null, value: null, originType: null, originId: null, element: null, originSlotIndex: null },
        // scherbius_did_encrypt: false, // This is part of latestServerState, not needed separately here for current round logic
        roundHistory: [],
        currentHistoryViewIndex: 0,
        latestServerState: null
    };
    const VIEWING_CURRENT_ROUND_INDEX = () => clientState.roundHistory.length;

    // --- API Calls --- (Same as before)
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
            updateGlobalUI(state); // Call the main UI updater
        } catch (error) {
            console.error(`API Error (${endpoint}):`, error);
            alert(`Error: ${error.message}`);
        }
    }

    // --- Main UI Update Function ---
    function updateGlobalUI(serverState) {
        console.log("Server state received:", serverState);
        gameAreaEl.style.display = 'block';

        turingScoreEl.textContent = serverState.turing_points;
        scherbiusScoreEl.textContent = "???"; // Scherbius points remain hidden
        maxVictoryPointsEl.textContent = serverState.max_victory_points;
        
        clientState.nBattles = serverState.n_battles;
        clientState.maxCardsPerBattle = serverState.max_cards_per_battle;
        clientState.roundHistory = serverState.round_history || [];
        clientState.latestServerState = serverState;

        if (serverState.is_game_over) {
            if (clientState.roundHistory.length > 0) {
                clientState.currentHistoryViewIndex = clientState.roundHistory.length - 1;
            } else {
                clientState.currentHistoryViewIndex = VIEWING_CURRENT_ROUND_INDEX();
            }
        } else {
            clientState.currentHistoryViewIndex = VIEWING_CURRENT_ROUND_INDEX();
        }

        if (serverState.current_phase === "Turing_Action" && clientState.currentHistoryViewIndex === VIEWING_CURRENT_ROUND_INDEX()) {
            clientState.initialHandForTurn = [...(serverState.turing_hand || [])]; 
            clientState.battlePlays = {};
            for(let i=0; i < clientState.nBattles; i++) clientState.battlePlays[`battle_${i}`] = [];
        }
        
        renderMainBattleDisplay(serverState); // Unified display function
        
        // REMOVED: Last round summary display logic
        // if (serverState.last_round_summary && clientState.currentHistoryViewIndex === VIEWING_CURRENT_ROUND_INDEX()) {
        //     lastRoundSummaryAreaEl.style.display = 'block';
        //     renderLastRoundSummary(serverState.last_round_summary); // This function will be removed
        // } else {
        //     lastRoundSummaryAreaEl.style.display = 'none';
        // }
        
        historicalRoundViewAreaEl.style.display = 'block';
        // historicalRoundContentEl.style.display = 'none'; // Content within this is not the primary battle display

        updateHistoryNavigationControls(); 
        manageRoundViewSpecificUI();

        if (serverState.is_game_over) {
            gameOverMessageEl.style.display = 'block';
            winnerEl.textContent = serverState.winner;
            submitTuringActionBtn.disabled = true;
        } else {
            gameOverMessageEl.style.display = 'none';
            submitTuringActionBtn.disabled = false;
        }
    }

    // --- Card Rendering and D&D --- (Largely same, createCardElement is fine)
    function createCardElement(cardObject, originType, originId, originSlotIndex = null) {
        const cardDiv = document.createElement('div');
        cardDiv.classList.add('card');
        cardDiv.textContent = cardObject.value; // Value could be 'X' if encrypted by backend
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
                value: e.target.dataset.cardValue === 'X' ? 'X' : parseInt(e.target.dataset.cardValue), // Handle 'X' for value if needed
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

    // findCardInArrayById, removeCardFromArrayById (Same)
    function findCardInArrayById(cardArray, cardId) {
        return cardArray.find(card => card && card.id === cardId);
    }
    function removeCardFromArrayById(cardArray, cardId) {
        const index = cardArray.findIndex(card => card && card.id === cardId);
        if (index > -1) return cardArray.splice(index, 1)[0];
        return null;
    }

    // renderAllPlayableCardAreas (Same)
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
    
    // handleCardDrop, handleCardDropToHand, addDropListenersToElement (Same)
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
        
        // Check if trying to add to a full battle, unless reordering within the same battle
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

        let cardObjectToMove = null;

        if (originType === 'battleSlot') {
            if (clientState.battlePlays[originId] && clientState.battlePlays[originId][originSlotIndex]) {
                cardObjectToMove = clientState.battlePlays[originId].splice(originSlotIndex, 1)[0];
            }
        }

        if (!cardObjectToMove) {
            console.error("Card to return to hand not found in origin slot.");
        }
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
                    renderAllPlayableCardAreas();
                    return;
                }
                handleCardDrop(battleId, slotIndex);
            }
        });
    }
    addDropListenersToElement(turingHandEl, true);


    // --- Main Battle Display (Unified for Current and Historical) ---
    function renderMainBattleDisplay(serverOrFullStateData) {
        const historyIdx = clientState.currentHistoryViewIndex;
        const isViewingCurrentRound = (historyIdx === VIEWING_CURRENT_ROUND_INDEX());
        
        rewardsDisplayEl.innerHTML = ''; 

        let numBattlesToDisplay, maxCardsPerBattleForDisplay;

        if (isViewingCurrentRound) {
            battleZoneTitleEl.textContent = "Current Battles & Rewards (Drag Your Cards Here)";
            numBattlesToDisplay = serverOrFullStateData.n_battles;
            maxCardsPerBattleForDisplay = clientState.maxCardsPerBattle;
        } else {
            const roundData = clientState.roundHistory[historyIdx];
            if (!roundData || !roundData.battles) {
                console.error("Historical round data or battles missing for index:", historyIdx);
                battleZoneTitleEl.textContent = "Error displaying historical round.";
                return;
            }
            battleZoneTitleEl.textContent = `Details for Past Round ${roundData.round_number}`;
            numBattlesToDisplay = roundData.battles.length;
            maxCardsPerBattleForDisplay = 0; // Not strictly needed for historical display of played cards
                                             // but kept if slot structure was to be mimicked
            if (roundData.battles.length > 0) {
                 maxCardsPerBattleForDisplay = roundData.battles.reduce((maxOverall, battle) => {
                    const maxInBattle = Math.max(
                        battle.turing_played_cards.length,
                        battle.scherbius_committed_cards.length
                    );
                    return Math.max(maxOverall, maxInBattle);
                }, 0);
            }
            if (maxCardsPerBattleForDisplay === 0) maxCardsPerBattleForDisplay = 3; // Fallback
        }

        if (isViewingCurrentRound) {
            const currentRewards = serverOrFullStateData.rewards;
            const currentScherbiusObserved = serverOrFullStateData.scherbius_observed_plays;
            // const currentScherbiusEncrypted = serverOrFullStateData.scherbius_did_encrypt; // Not used for styling

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
                    rewardInfoHTML += `<p>Potential Rewards: <div class="display-card-container"></div></p>`; // Empty container
                }
                rewardInfoHTML += `</div>`;

                let scherbiusInfoHTML = `<div class="scherbius-observed-info">`;
                const scherbiusPlayForBattle = (currentScherbiusObserved && currentScherbiusObserved[i]) ? currentScherbiusObserved[i] : [];
                let scherbiusCardsContent = '<div class="display-card-container">';
                if (scherbiusPlayForBattle.length > 0) {
                    scherbiusPlayForBattle.forEach(cardVal => { // cardVal is 'X' if encrypted by backend
                        const cardEl = document.createElement('div');
                        cardEl.classList.add('display-card', 'scherbius-card');
                        // REMOVED: No scherbius-card-encrypted class based on flag
                        cardEl.textContent = cardVal;
                        scherbiusCardsContent += cardEl.outerHTML;
                    });
                } // else: scherbiusCardsContent remains an empty container
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
            if (!roundData) return;

            roundData.battles.forEach(battle => {
                const battleDiv = document.createElement('div');
                battleDiv.classList.add('battle-item', 'historical-battle-item');

                // Battle Title
                let titleHTML = `<h4>Battle ${battle.id + 1} (Round ${roundData.round_number})</h4>`;

                // Battle Outcome
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
                
                // Historical Reward Info
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
                // if (!hasRewards) rewardsContent += ''; // Empty container is default
                rewardsContent += '</div>';
                rewardInfoHTML += `<p>Rewards Available: ${rewardsContent}</p></div>`; // Changed label slightly

                // Scherbius Committed Info
                let scherbiusInfoHTML = `<div class="scherbius-observed-info">`; // Re-using class for layout consistency
                let scherbiusCardsContent = '<div class="display-card-container">';
                if (battle.scherbius_committed_cards.length > 0) {
                    battle.scherbius_committed_cards.forEach(cardVal => { // cardVal is 'X' if encrypted by backend
                        const cardEl = document.createElement('div');
                        cardEl.classList.add('display-card', 'scherbius-card');
                        // REMOVED: No scherbius-card-encrypted class based on flag
                        cardEl.textContent = cardVal;
                        scherbiusCardsContent += cardEl.outerHTML;
                    });
                } // else: scherbiusCardsContent remains an empty container
                scherbiusCardsContent += '</div>';
                scherbiusInfoHTML += `<p>Scherbius Committed: ${scherbiusCardsContent}</p></div>`;
                
                // Turing's Played Cards
                let turingPlayedHTML = `<div class="turing-played-cards-area static-played-cards-area"><h4>Turing Played:</h4>`;
                let turingCardsContent = '<div class="display-card-container">';
                if (battle.turing_played_cards.length > 0) {
                    battle.turing_played_cards.forEach(cardVal => {
                         turingCardsContent += `<div class="display-card turing-summary-card">${cardVal}</div>`;
                    });
                } // else: turingCardsContent remains an empty container
                turingCardsContent += '</div>';
                turingPlayedHTML += `${turingCardsContent}</div>`;

                battleDiv.innerHTML = titleHTML + battleOutcomeHTML + rewardInfoHTML + scherbiusInfoHTML + turingPlayedHTML;
                rewardsDisplayEl.appendChild(battleDiv);
            });
        }
    }
    
    // REMOVED: renderLastRoundSummary function
    // function renderLastRoundSummary(summary) { ... }

    // --- Event Listeners ---
    newGameBtn.addEventListener('click', () => {
        // REMOVED: lastRoundSummaryAreaEl.style.display = 'none';
        gameOverMessageEl.style.display = 'none';
        clientState.roundHistory = [];
        // currentHistoryViewIndex will be set by updateGlobalUI
        fetchApi('/new_game', 'POST');
    });

    submitTuringActionBtn.addEventListener('click', () => {
        const finalTuringStrategyValues = [];
        for (let i = 0; i < clientState.nBattles; i++) {
            const battleId = `battle_${i}`;
            const cardsInBattle = clientState.battlePlays[battleId] || [];
            finalTuringStrategyValues.push(cardsInBattle.map(cardObj => {
                // Ensure card value is correctly parsed if it was 'X' or similar
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
    // REMOVED: lastRoundSummaryAreaEl.style.display = 'none';
    // historicalRoundContentEl.style.display = 'none'; // Handled by manageRoundViewSpecificUI or direct styling
    clientState.currentHistoryViewIndex = VIEWING_CURRENT_ROUND_INDEX();
    updateHistoryNavigationControls();
    manageRoundViewSpecificUI();

    // --- UI State Management for Current vs. Historical View ---
    function manageRoundViewSpecificUI() {
        const isViewingCurrent = clientState.currentHistoryViewIndex === VIEWING_CURRENT_ROUND_INDEX();
        const isGameOver = clientState.latestServerState ? clientState.latestServerState.is_game_over : true; 
        // const hasLastRoundSummary = clientState.latestServerState && clientState.latestServerState.last_round_summary; // Not needed

        turingHandEl.style.display = isViewingCurrent ? 'flex' : 'none';
        document.getElementById('turingHandArea').style.display = isViewingCurrent ? 'block' : 'none'; // Show/hide whole hand area
        document.getElementById('turingActionControls').querySelector('h3').style.display = isViewingCurrent ? 'block' : 'none'; // Show/hide "Your Hand:" title


        submitTuringActionBtn.style.display = isViewingCurrent ? 'block' : 'none'; // Show/hide submit button
        submitTuringActionBtn.disabled = !isViewingCurrent || isGameOver;
        
        // REMOVED: Last round summary visibility logic
        // if (isViewingCurrent && hasLastRoundSummary) { ... }

        gameAreaEl.style.display = clientState.latestServerState ? 'block' : 'none'; 
        historicalRoundViewAreaEl.style.display = clientState.latestServerState ? 'block' : 'none';
        // The content of historicalRoundViewArea (like historicalRoundInfoEl) is not directly managed here for battle display
        // as renderMainBattleDisplay populates rewardsDisplayEl.
        // document.getElementById('historicalRoundContent').style.display = isViewingCurrent ? 'none' : 'block'; // Example if it had separate content
    }

    // --- Navigation Controls Update --- (Same as before)
    function updateHistoryNavigationControls() {
        const historyLen = clientState.roundHistory.length;
        const viewingCurrent = clientState.currentHistoryViewIndex === VIEWING_CURRENT_ROUND_INDEX();

        prevRoundBtnEl.disabled = clientState.currentHistoryViewIndex <= 0;
        nextRoundBtnEl.disabled = viewingCurrent;

        if (viewingCurrent) {
            if (historyLen > 0) {
                historicalRoundIndicatorEl.textContent = `Viewing Current Round (After Round ${historyLen})`;
            } else {
                historicalRoundIndicatorEl.textContent = "Viewing Current Round (Round 1)";
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
            renderMainBattleDisplay(clientState.latestServerState);
            updateHistoryNavigationControls();
            manageRoundViewSpecificUI();
        }
    });
});