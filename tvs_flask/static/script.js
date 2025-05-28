document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements --- (Same as before)
    const turingScoreEl = document.getElementById('turingScore');
    const scherbiusScoreEl = document.getElementById('scherbiusScore');
    const maxVictoryPointsEl = document.getElementById('maxVictoryPoints');
    const winnerEl = document.getElementById('winner');
    const newGameBtn = document.getElementById('newGameBtn');
    const submitTuringActionBtn = document.getElementById('submitTuringActionBtn');
    const gameAreaEl = document.getElementById('gameArea');
    const gameOverMessageEl = document.getElementById('gameOverMessage');
    const lastRoundSummaryAreaEl = document.getElementById('lastRoundSummaryArea');
    const lastRoundInfoEl = document.getElementById('lastRoundInfo');
    const lastRoundSummaryBattlesEl = document.getElementById('lastRoundSummaryBattles');
    const rewardsDisplayEl = document.getElementById('rewardsDisplay');
    const turingHandEl = document.getElementById('turingHand');

    // --- DOM Elements for Historical View ---
    const historicalRoundViewAreaEl = document.getElementById('historicalRoundViewArea');
    const prevRoundBtnEl = document.getElementById('prevRoundBtn');
    const nextRoundBtnEl = document.getElementById('nextRoundBtn');
    const historicalRoundIndicatorEl = document.getElementById('historicalRoundIndicator');
    const historicalRoundContentEl = document.getElementById('historicalRoundContent'); // Parent for info + battles
    const historicalRoundInfoEl = document.getElementById('historicalRoundInfo');
    const historicalRoundBattlesEl = document.getElementById('historicalRoundBattles');


    // --- Client-Side Game State ---
    let clientState = {
        initialHandForTurn: [], currentHandDisplayObjects: [],
        battlePlays: {},
        nBattles: 0,
        maxCardsPerBattle: 0,
        draggedCard: { id: null, value: null, originType: null, originId: null, element: null, originSlotIndex: null },
        scherbius_did_encrypt: false, 
        roundHistory: [],
        // currentHistoryViewIndex:
        // 0 to roundHistory.length - 1 for historical rounds.
        // roundHistory.length for the current, interactive round.
        currentHistoryViewIndex: 0, // Default to current round view initially (will be set by history length)
        latestServerState: null // To cache the latest server state for rendering current round
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
            updateGlobalUI(state);
        } catch (error) {
            console.error(`API Error (${endpoint}):`, error);
            alert(`Error: ${error.message}`);
        }
    }

    // --- Main UI Update Function --- (Same, ensures maxCardsPerBattle is set)
    function updateGlobalUI(serverState) {
        console.log("Server state received:", serverState);
        gameAreaEl.style.display = 'block';

        turingScoreEl.textContent = serverState.turing_points;
        scherbiusScoreEl.textContent = "???";
        maxVictoryPointsEl.textContent = serverState.max_victory_points;
        
        clientState.nBattles = serverState.n_battles;
        clientState.maxCardsPerBattle = serverState.max_cards_per_battle;
        // Store scherbius_did_encrypt in clientState for easier access in renderRewardsAndBattleDropZones
        // This is assuming scherbius_did_encrypt is part of the serverState structure at this level
        clientState.scherbius_did_encrypt = serverState.scherbius_did_encrypt; // For current round encryption status
        clientState.roundHistory = serverState.round_history || [];
        clientState.latestServerState = serverState; // Cache the full server state

        // On new data, typically view the current round, unless game is over.
        // If game is over, try to view the last historical round.
        // If no history and game over, currentHistoryViewIndex might become 0, but VIEWING_CURRENT_ROUND_INDEX() would also be 0.
        if (serverState.is_game_over) {
            if (clientState.roundHistory.length > 0) {
                clientState.currentHistoryViewIndex = clientState.roundHistory.length - 1; // View last historical round
            } else {
                // No history, game over (e.g. game over on first round) - show current (empty) state
                clientState.currentHistoryViewIndex = VIEWING_CURRENT_ROUND_INDEX();
            }
        } else {
             // Default to viewing the current round if game is not over
            clientState.currentHistoryViewIndex = VIEWING_CURRENT_ROUND_INDEX();
        }


        if (serverState.current_phase === "Turing_Action" && clientState.currentHistoryViewIndex === VIEWING_CURRENT_ROUND_INDEX()) {
            clientState.initialHandForTurn = [...(serverState.turing_hand || [])]; 
            clientState.battlePlays = {}; // This will store arrays of card objects
            for(let i=0; i < clientState.nBattles; i++) clientState.battlePlays[`battle_${i}`] = [];
        }
        
        renderMainBattleDisplay(serverState); // Unified display function
        // renderAllPlayableCardAreas() is called within renderMainBattleDisplay if viewing current round
        
        // Manage last round summary visibility based on current view
        if (serverState.last_round_summary && clientState.currentHistoryViewIndex === VIEWING_CURRENT_ROUND_INDEX()) {
            lastRoundSummaryAreaEl.style.display = 'block';
            renderLastRoundSummary(serverState.last_round_summary);
        } else {
            lastRoundSummaryAreaEl.style.display = 'none';
        }
        
        // Old historical view area is no longer the primary display, hide it.
        // Its navigation controls (prev/next buttons, indicator) are now repurposed.
        historicalRoundViewAreaEl.style.display = 'block'; // Keep the container for nav buttons visible
        historicalRoundContentEl.style.display = 'none'; // Hide the specific content part of old historical view

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

    // --- Card Rendering and D&D ---
    function createCardElement(cardObject, originType, originId, originSlotIndex = null) { // Added originSlotIndex
        const cardDiv = document.createElement('div');
        cardDiv.classList.add('card');
        cardDiv.textContent = cardObject.value;
        cardDiv.draggable = true;
        cardDiv.dataset.cardId = cardObject.id;
        cardDiv.dataset.cardValue = cardObject.value;
        cardDiv.dataset.originType = originType; // 'hand', 'battleSlot'
        cardDiv.dataset.originId = originId;     // 'turingHand', 'battle_X' (battle area ID)
        if (originSlotIndex !== null) {
            cardDiv.dataset.originSlotIndex = originSlotIndex; // Index of the slot within the battle area
        }


        cardDiv.addEventListener('dragstart', (e) => {
            clientState.draggedCard = { 
                id: e.target.dataset.cardId, 
                value: parseInt(e.target.dataset.cardValue), 
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
        // 1. Determine cards effectively in hand
        let cardsEffectivelyInHandObjects = [...clientState.initialHandForTurn];
        Object.values(clientState.battlePlays).flat().forEach(playedCardObj => {
            if (playedCardObj) { // Filter out nulls if any (though battlePlays should be dense arrays of card objects)
                cardsEffectivelyInHandObjects = cardsEffectivelyInHandObjects.filter(handCardObj => handCardObj.id !== playedCardObj.id);
            }
        });
        clientState.currentHandDisplayObjects = cardsEffectivelyInHandObjects;

        // 2. Render hand
        turingHandEl.innerHTML = '';
        clientState.currentHandDisplayObjects.forEach(cardObj => {
            turingHandEl.appendChild(createCardElement(cardObj, 'hand', 'turingHand'));
        });

        // 3. Render cards into battle slots
        for (let i = 0; i < clientState.nBattles; i++) {
            const battleId = `battle_${i}`;
            const battleAreaEl = rewardsDisplayEl.querySelector(`.turing-played-cards-area[data-battle-id="${battleId}"]`);
            if (battleAreaEl) {
                const slots = battleAreaEl.querySelectorAll('.card-slot');
                slots.forEach(slot => slot.innerHTML = ''); // Clear existing cards from slots

                const cardsInThisBattle = clientState.battlePlays[battleId] || [];
                cardsInThisBattle.forEach((cardObj, slotIndex) => {
                    if (cardObj && slotIndex < slots.length) { // Ensure card exists and slot exists
                        slots[slotIndex].appendChild(createCardElement(cardObj, 'battleSlot', battleId, slotIndex));
                    }
                });
            }
        }
    }
    
    function handleCardDrop(targetBattleId, targetSlotIndex) { // targetId is now battleId, targetSlotIndex is new
        const { id: draggedCardId, originType, originId, originSlotIndex } = clientState.draggedCard;
        if (!draggedCardId) return;

        let cardObjectToMove = null;

        // --- Remove card from its origin ---
        if (originType === 'hand') {
            cardObjectToMove = findCardInArrayById(clientState.initialHandForTurn, draggedCardId);
            // Don't remove from initialHandForTurn; it's the master list for the turn.
            // It will be filtered out by renderAllPlayableCardAreas if successfully placed in a battle.
        } else if (originType === 'battleSlot') {
            // Card is coming from another battle slot (or the same one)
            if (clientState.battlePlays[originId] && clientState.battlePlays[originId][originSlotIndex]) {
                // To "remove" from a specific slot, we need to update the battlePlays array.
                // We'll effectively nullify it here, and then compact the array later or handle nulls.
                // A simpler way: just remove it from the array for that battle.
                // The card object itself is what we need.
                cardObjectToMove = clientState.battlePlays[originId][originSlotIndex];
                clientState.battlePlays[originId].splice(originSlotIndex, 1); // Remove from old slot array
            }
        }

        if (!cardObjectToMove) {
            console.error("Dragged card object not found or couldn't be removed from origin!", clientState.draggedCard);
            renderAllPlayableCardAreas();
            return;
        }

        // --- Place card into the target slot ---
        // targetBattleId is the ID of the battle area (e.g., "battle_0")
        // targetSlotIndex is the index of the slot within that battle area

        const targetBattleCards = clientState.battlePlays[targetBattleId];
        if (!targetBattleCards) { // Should not happen if drop zones are correctly set up
            console.error("Target battle play array not found:", targetBattleId);
            // Attempt to return card to origin if it was from a battle slot
            if (originType === 'battleSlot') clientState.battlePlays[originId].splice(originSlotIndex, 0, cardObjectToMove);
            renderAllPlayableCardAreas();
            return;
        }

        // Check if the target slot is already occupied by a DIFFERENT card
        // Or if the battle is already full (though individual slot drop targets make this less of an issue for the specific slot)
        if (targetBattleCards.length >= clientState.maxCardsPerBattle && 
            !(originType === 'battleSlot' && originId === targetBattleId)) { // Allow reordering within the same battle if full
            console.warn(`Battle ${targetBattleId} is full. Cannot add new card.`);
             // Attempt to return card to origin if it was from a battle slot
            if (originType === 'battleSlot') clientState.battlePlays[originId].splice(originSlotIndex, 0, cardObjectToMove);
            renderAllPlayableCardAreas();
            return;
        }
        
        // If dropping onto a specific slot, and that slot is occupied by another card,
        // we might want to swap or prevent. For now, let's assume we are adding to the array
        // and the visual rendering will place it in the first available empty DOM slot,
        // or we directly manage which card object goes to which slot index in battlePlays.

        // Let's refine: clientState.battlePlays[battleId] should be an array representing the cards in order of their slots.
        // If a slot is empty, we can represent it with `null` or just have a shorter array.
        // For simplicity, let's keep it as a dense array of card objects.
        // When dropping, we add to this array. renderAllPlayableCardAreas will fill the DOM slots in order.

        // Add the card to the target battle's array of cards
        // We need to ensure we don't duplicate if dragging within the same battle area
        if (originType === 'battleSlot' && originId === targetBattleId) {
            // Reordering within the same battle: card was already removed, now add it back at new position
            // For simplicity, just add to end, sorting/specific slot placement can be complex.
            // Or, if targetSlotIndex is reliable, insert there.
            // Let's assume for now we just add to the list for that battle.
            // The visual update will re-render them in order.
            targetBattleCards.splice(targetSlotIndex, 0, cardObjectToMove); // Insert at specific slot index
        } else {
            // Coming from hand or different battle
            // Add to the target battle's list of cards, respecting maxCardsPerBattle
            if (targetBattleCards.length < clientState.maxCardsPerBattle) {
                targetBattleCards.splice(targetSlotIndex, 0, cardObjectToMove); // Insert at specific slot index
            } else {
                 // This case should be caught by the earlier check, but as a safeguard:
                console.warn("Target battle is full, cannot place card (safeguard).");
                if (originType === 'battleSlot') clientState.battlePlays[originId].splice(originSlotIndex, 0, cardObjectToMove); // Return to origin
            }
        }
        
        // Compact the array for any nulls if we were using them (not in this simplified version)
        clientState.battlePlays[targetBattleId] = clientState.battlePlays[targetBattleId].filter(c => c !== null);


        renderAllPlayableCardAreas();
    }
    
    // Modified to handle card drop onto the hand area
    function handleCardDropToHand() {
        const { id: draggedCardId, originType, originId, originSlotIndex } = clientState.draggedCard;
        if (!draggedCardId || originType === 'hand') { // No action if dragging from hand to hand
            renderAllPlayableCardAreas(); // Just re-render to remove dragging class
            return;
        }

        let cardObjectToMove = null;

        if (originType === 'battleSlot') {
            if (clientState.battlePlays[originId] && clientState.battlePlays[originId][originSlotIndex]) {
                cardObjectToMove = clientState.battlePlays[originId].splice(originSlotIndex, 1)[0]; // Remove and get card
            }
        }

        if (!cardObjectToMove) {
            console.error("Card to return to hand not found in origin slot.");
            renderAllPlayableCardAreas();
            return;
        }
        // Card is now "removed" from battlePlays.
        // renderAllPlayableCardAreas will automatically include it in the hand display
        // because it's in initialHandForTurn but no longer in any battlePlays.
        renderAllPlayableCardAreas();
    }


    function addDropListenersToElement(element, isHandArea, battleId = null, slotIndex = null) {
        element.addEventListener('dragover', e => { 
            e.preventDefault(); 
            if (!isHandArea) { // It's a battle slot
                const targetBattleCards = clientState.battlePlays[battleId] || [];
                // Allow drop if slot is empty OR if dragging from another slot in the SAME battle (reordering)
                const cardInSlot = element.querySelector('.card');
                if (cardInSlot && !(clientState.draggedCard.originType === 'battleSlot' && clientState.draggedCard.originId === battleId)) {
                    element.classList.add('drag-over-full'); // Slot is occupied by a different card
                } else if (targetBattleCards.length >= clientState.maxCardsPerBattle && 
                           !(clientState.draggedCard.originType === 'battleSlot' && clientState.draggedCard.originId === battleId) &&
                           !cardInSlot) { // Battle is full and trying to drop into an empty slot (shouldn't happen if slots = max)
                     element.classList.add('drag-over-full'); // Battle area is full
                } else {
                    element.classList.add('drag-over');
                }
            } else { // It's the hand area
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
                // Check if the slot is already occupied by a card NOT being dragged
                const existingCardElement = element.querySelector('.card');
                if (existingCardElement && existingCardElement !== clientState.draggedCard.element) {
                    console.warn("Slot is already occupied by a different card.");
                    // Potentially trigger a swap or just prevent drop and re-render
                    renderAllPlayableCardAreas();
                    return;
                }
                handleCardDrop(battleId, slotIndex);
            }
        });
    }
    addDropListenersToElement(turingHandEl, true); // Listener for the main hand area

    // --- Main Battle Display (Unified for Current and Historical) ---
    function renderMainBattleDisplay(serverOrFullStateData) {
        // serverOrFullStateData is the full state from updateGlobalUI (clientState.latestServerState)
        // when called from navigation, or the direct serverState on first load.
        const historyIdx = clientState.currentHistoryViewIndex;
        const isViewingCurrentRound = (historyIdx === VIEWING_CURRENT_ROUND_INDEX());
        
        rewardsDisplayEl.innerHTML = ''; // Clear previous content

        let numBattlesToDisplay, maxCardsPerBattleForDisplay;

        if (isViewingCurrentRound) {
            numBattlesToDisplay = serverOrFullStateData.n_battles;
            maxCardsPerBattleForDisplay = clientState.maxCardsPerBattle; // From global clientState for current round
        } else {
            const roundData = clientState.roundHistory[historyIdx];
            if (!roundData || !roundData.battles) {
                console.error("Historical round data or battles missing for index:", historyIdx);
                return; // Or display an error message
            }
            numBattlesToDisplay = roundData.battles.length;
            // Calculate max cards based on actual historical data for that round
            maxCardsPerBattleForDisplay = 0;
            if (roundData.battles.length > 0) {
                 maxCardsPerBattleForDisplay = roundData.battles.reduce((maxOverall, battle) => {
                    const maxInBattle = Math.max(
                        battle.turing_played_cards.length,
                        battle.scherbius_committed_cards.length
                        // We don't have slot structure in historical, so max of played cards is best guess
                    );
                    return Math.max(maxOverall, maxInBattle);
                }, 0);
            }
             // If for some reason all historical battles had 0 cards, default to a reasonable number like 1 or 3.
            if (maxCardsPerBattleForDisplay === 0) maxCardsPerBattleForDisplay = 3;
        }


        if (isViewingCurrentRound) {
            // RENDER CURRENT INTERACTIVE ROUND
            const currentRewards = serverOrFullStateData.rewards;
            const currentScherbiusObserved = serverOrFullStateData.scherbius_observed_plays;
            const currentScherbiusEncrypted = serverOrFullStateData.scherbius_did_encrypt;

            for (let i = 0; i < numBattlesToDisplay; i++) {
                const battleId = `battle_${i}`;
                const battleDiv = document.createElement('div');
                battleDiv.classList.add('battle-item');

                // Reward Info (current round)
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
                    rewardInfoHTML += `<p>Potential Rewards: <span class="no-reward-text">None</span></p>`;
                }
                rewardInfoHTML += `</div>`;

                // Scherbius Observed Info (current round)
                let scherbiusInfoHTML = `<div class="scherbius-observed-info">`;
                const scherbiusPlayForBattle = (currentScherbiusObserved && currentScherbiusObserved[i]) ? currentScherbiusObserved[i] : [];
                let scherbiusCardsContent = '<div class="display-card-container">';
                if (scherbiusPlayForBattle.length > 0) {
                    scherbiusPlayForBattle.forEach(cardVal => {
                        const cardEl = document.createElement('div');
                        cardEl.classList.add('display-card', 'scherbius-card');
                        if (currentScherbiusEncrypted) {
                            cardEl.classList.add('scherbius-card-encrypted');
                        }
                        cardEl.textContent = cardVal;
                        scherbiusCardsContent += cardEl.outerHTML;
                    });
                } else { scherbiusCardsContent += '<span class="no-play-text">None</span>'; }
                scherbiusCardsContent += '</div>';
                scherbiusInfoHTML += `<p>Scherbius Will Play: ${scherbiusCardsContent}</p></div>`;

                // Turing's Play Area (ACTIVE Drop Zones)
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
            renderAllPlayableCardAreas(); // Populate active slots with cards from clientState.battlePlays

        } else {
            // RENDER HISTORICAL ROUND (STATIC)
            const roundData = clientState.roundHistory[historyIdx];
            if (!roundData) return; // Should not happen if historyIdx is valid

            roundData.battles.forEach(battle => {
                const battleDiv = document.createElement('div');
                battleDiv.classList.add('battle-item', 'historical-battle-item');

                // Historical Reward Info
                let rewardInfoHTML = `<div class="reward-info"><h4>Battle ${battle.id + 1} (Round ${roundData.round_number})</h4>`;
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
                if (!hasRewards) rewardsContent += '<span class="no-reward-text">None</span>';
                rewardsContent += '</div>';
                rewardInfoHTML += `<p>Rewards Turing Could Have Won: ${rewardsContent}</p></div>`;

                // Scherbius Committed Info (past round)
                let scherbiusInfoHTML = `<div class="scherbius-observed-info">`;
                let scherbiusCardsContent = '<div class="display-card-container">';
                if (battle.scherbius_committed_cards.length > 0) {
                    battle.scherbius_committed_cards.forEach(cardVal => {
                        const cardEl = document.createElement('div');
                        cardEl.classList.add('display-card', 'scherbius-card');
                        if (roundData.scherbius_encrypted_this_round) {
                            cardEl.classList.add('scherbius-card-encrypted');
                        }
                        cardEl.textContent = cardVal;
                        scherbiusCardsContent += cardEl.outerHTML;
                    });
                } else { scherbiusCardsContent += '<span class="no-play-text">None</span>'; }
                scherbiusCardsContent += '</div>';
                scherbiusInfoHTML += `<p>Scherbius Committed: ${scherbiusCardsContent}</p></div>`;
                
                // Turing's Played Cards (STATIC from past round)
                let turingPlayedHTML = `<div class="turing-played-cards-area static-played-cards-area"><h4>Turing Played:</h4>`;
                let turingCardsContent = '<div class="display-card-container">';
                if (battle.turing_played_cards.length > 0) {
                    battle.turing_played_cards.forEach(cardVal => {
                         turingCardsContent += `<div class="display-card turing-summary-card">${cardVal}</div>`;
                    });
                } else { turingCardsContent += '<span class="no-play-text">None</span>'; }
                turingCardsContent += '</div>';
                turingPlayedHTML += `${turingCardsContent}</div>`;

                battleDiv.innerHTML = rewardInfoHTML + scherbiusInfoHTML + turingPlayedHTML;
                rewardsDisplayEl.appendChild(battleDiv);
            });
        }
    }
    
    function renderLastRoundSummary(summary) {
        let infoText = `Points Gained This Round - Turing: ${summary.turing_points_gained_in_round}, Scherbius: ${summary.scherbius_points_gained_in_round}. `;
        if (summary.scherbius_reencrypted_this_round) { // Assuming you add this flag from backend
            infoText += ` Scherbius re-encrypted.`;
        }
        lastRoundInfoEl.innerHTML = infoText;

        lastRoundSummaryBattlesEl.innerHTML = ''; // Clear previous summary
        summary.battle_details.forEach(battle => {
            const reportDiv = document.createElement('div');
            // Use a class consistent with your CSS for summary battle reports
            // e.g., .summary-battle-report if that's what you have
            reportDiv.classList.add('summary-battle-report'); 

            let turingPlayedHTML = '<div class="display-card-container">';
            if (battle.turing_played.length > 0) {
                battle.turing_played.forEach(cardVal => {
                    turingPlayedHTML += `<div class="display-card turing-summary-card">${cardVal}</div>`;
                });
            } else {
                turingPlayedHTML += '<span class="no-play-text">None</span>';
            }
            turingPlayedHTML += '</div>';

            let scherbiusCommittedHTML = '<div class="display-card-container">';
            if (battle.scherbius_committed.length > 0) {
                battle.scherbius_committed.forEach(cardVal => {
                    scherbiusCommittedHTML += `<div class="display-card scherbius-card">${cardVal}</div>`;
                });
            } else {
                scherbiusCommittedHTML += '<span class="no-play-text">None</span>';
            }
            scherbiusCommittedHTML += '</div>';
            
            // Determine battle winner text/style
            let battleOutcomeText = '';
            if (battle.winner === 'Turing') {
                battleOutcomeText = `<p class="battle-outcome turing-wins">Turing won this battle.</p>`;
            } else if (battle.winner === 'Scherbius') {
                battleOutcomeText = `<p class="battle-outcome scherbius-wins">Scherbius won this battle.</p>`;
            } else if (battle.winner === 'Draw') { // Assuming 'Draw' or null/None for no winner
                battleOutcomeText = `<p class="battle-outcome draw">Battle was a draw.</p>`;
            }


            reportDiv.innerHTML = `
                <h5>Battle ${battle.battle_id + 1}</h5>
                ${battleOutcomeText}
                <p>Turing Played: ${turingPlayedHTML}</p>
                <p>Scherbius Committed: ${scherbiusCommittedHTML}</p>
            `;
            lastRoundSummaryBattlesEl.appendChild(reportDiv);
        });
    }

    // --- Event Listeners (Submit button needs to collect cards correctly) ---
    newGameBtn.addEventListener('click', () => {
        lastRoundSummaryAreaEl.style.display = 'none';
        gameOverMessageEl.style.display = 'none';
        clientState.roundHistory = [];
        clientState.currentHistoryViewIndex = VIEWING_CURRENT_ROUND_INDEX(); // Set to current round view
        // renderMainBattleDisplay will be called by updateGlobalUI after fetch
        // updateHistoryNavigationControls will be called by updateGlobalUI
        // manageRoundViewSpecificUI will be called by updateGlobalUI
        fetchApi('/new_game', 'POST');
    });

    submitTuringActionBtn.addEventListener('click', () => {
        const finalTuringStrategyValues = [];
        for (let i = 0; i < clientState.nBattles; i++) {
            const battleId = `battle_${i}`;
            // Get cards from clientState.battlePlays for this battleId
            const cardsInBattle = clientState.battlePlays[battleId] || [];
            finalTuringStrategyValues.push(cardsInBattle.map(cardObj => cardObj.value));
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
    lastRoundSummaryAreaEl.style.display = 'none';
    // historicalRoundViewAreaEl itself is kept for nav buttons, but content is hidden
    historicalRoundContentEl.style.display = 'none'; 
    clientState.currentHistoryViewIndex = VIEWING_CURRENT_ROUND_INDEX(); // Start by viewing current (empty)
    updateHistoryNavigationControls(); // Initialize button states
    manageRoundViewSpecificUI();


    // --- UI State Management for Current vs. Historical View ---
    function manageRoundViewSpecificUI() {
        const isViewingCurrent = clientState.currentHistoryViewIndex === VIEWING_CURRENT_ROUND_INDEX();
        
        // Default values for when latestServerState might be null (e.g., initial load)
        const isGameOver = clientState.latestServerState ? clientState.latestServerState.is_game_over : true; 
        const hasLastRoundSummary = clientState.latestServerState && clientState.latestServerState.last_round_summary;

        turingHandEl.style.display = isViewingCurrent ? 'flex' : 'none'; // Hand is flex
        submitTuringActionBtn.disabled = !isViewingCurrent || isGameOver;
        
        // Last round summary is only shown if viewing the current round AND there is a summary to show
        if (isViewingCurrent && hasLastRoundSummary) {
            lastRoundSummaryAreaEl.style.display = 'block';
            // renderLastRoundSummary is called from updateGlobalUI
        } else {
            lastRoundSummaryAreaEl.style.display = 'none';
        }

        // The main gameArea (which contains rewardsDisplayEl) is always visible
        // unless the game hasn't started. The content of rewardsDisplayEl changes.
        // gameAreaEl contains rewardsDisplayEl and turingActionControls (hand)
        gameAreaEl.style.display = clientState.latestServerState ? 'block' : 'none'; 
        
        // The navigation area itself should always be visible if history is potentially available or game started
        historicalRoundViewAreaEl.style.display = clientState.latestServerState ? 'block' : 'none';
        // historicalRoundContentEl (the specific content part of the old historical view, like historicalRoundInfoEl) is not used for battle display.
        // It's hidden in updateGlobalUI and its elements are not populated by renderMainBattleDisplay.
        // If historicalRoundInfoEl were to be used for general round info outside battle items, that would be a separate display update.
        // For now, all round specific info is within the battle items or the indicator.
        historicalRoundContentEl.style.display = 'none'; 
    }


    // --- Navigation Controls Update ---
    function updateHistoryNavigationControls() {
        const historyLen = clientState.roundHistory.length;
        const viewingCurrent = clientState.currentHistoryViewIndex === VIEWING_CURRENT_ROUND_INDEX();

        prevRoundBtnEl.disabled = clientState.currentHistoryViewIndex <= 0; // Disabled if viewing oldest history or no history
        nextRoundBtnEl.disabled = viewingCurrent; // Disabled if viewing current round

        if (viewingCurrent) {
            if (historyLen > 0) {
                historicalRoundIndicatorEl.textContent = `Viewing Current Round (After Round ${historyLen})`;
            } else {
                historicalRoundIndicatorEl.textContent = "Viewing Current Round (Round 1)";
            }
        } else {
            // Viewing a historical round
            if (historyLen > 0 && clientState.currentHistoryViewIndex < historyLen) {
                historicalRoundIndicatorEl.textContent = `Viewing Past Round ${clientState.roundHistory[clientState.currentHistoryViewIndex].round_number} of ${historyLen}`;
            } else {
                 // Should not happen if logic is correct, but as a fallback:
                historicalRoundIndicatorEl.textContent = "History Navigation";
            }
        }
    }

    prevRoundBtnEl.addEventListener('click', () => {
        if (clientState.currentHistoryViewIndex > 0) {
            clientState.currentHistoryViewIndex--;
            renderMainBattleDisplay(clientState.latestServerState); // Re-render main display
            updateHistoryNavigationControls();
            manageRoundViewSpecificUI();
        }
    });

    nextRoundBtnEl.addEventListener('click', () => {
        if (clientState.currentHistoryViewIndex < VIEWING_CURRENT_ROUND_INDEX()) {
            clientState.currentHistoryViewIndex++;
            renderMainBattleDisplay(clientState.latestServerState); // Re-render main display
            updateHistoryNavigationControls();
            manageRoundViewSpecificUI();
        }
    });

    // Remove or comment out the old renderHistoricalRound function as its functionality
    // for displaying battles is now merged into renderMainBattleDisplay.
    // The elements historicalRoundInfoEl and historicalRoundBattlesEl are also no longer directly used.
    // function renderHistoricalRound(roundIndex) { ... }
});