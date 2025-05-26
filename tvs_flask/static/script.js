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

    // --- Client-Side Game State --- (Same as before)
    let clientState = {
        initialHandForTurn: [], currentHandDisplayObjects: [],
        battlePlays: {},
        nBattles: 0,
        maxCardsPerBattle: 0,
        draggedCard: { id: null, value: null, originType: null, originId: null, element: null, originSlotIndex: null } // Added originSlotIndex
    };

    // --- API Calls (Same) ---
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

        if (serverState.current_phase === "Turing_Action") {
            clientState.initialHandForTurn = [...(serverState.turing_hand || [])]; 
            clientState.battlePlays = {}; // This will store arrays of card objects
            for(let i=0; i < clientState.nBattles; i++) clientState.battlePlays[`battle_${i}`] = [];
        }
        
        renderRewardsAndBattleDropZones(serverState.rewards, serverState.n_battles, serverState.scherbius_observed_plays); // Render drop zones first
        renderAllPlayableCardAreas(); // Then render cards into them and hand
        
        if (serverState.last_round_summary) {
            lastRoundSummaryAreaEl.style.display = 'block';
            renderLastRoundSummary(serverState.last_round_summary);
        } else {
            lastRoundSummaryAreaEl.style.display = 'none';
        }

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

    function renderRewardsAndBattleDropZones(rewardsData, numBattles, scherbiusObservedPlays) {
        rewardsDisplayEl.innerHTML = ''; // Clear previous battle zones
        for (let i = 0; i < numBattles; i++) {
            const battleId = `battle_${i}`;
            const battleDiv = document.createElement('div');
            battleDiv.classList.add('battle-item'); // This class is from your original CSS for battle items
                                                    // If you used .interactive-battle-container .battle-item, ensure this matches.
                                                    // For consistency, let's assume .battle-item is the primary class.

            // --- Reward Info ---
            let rewardInfoHTML = `<div class="reward-info"><h4>Battle ${i + 1}</h4>`;
            const vpReward = rewardsData.vp_rewards[i];
            const cardRewardArray = rewardsData.card_rewards[i];
            
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
                rewardInfoHTML += `<p>Rewards: ${rewardsContent}</p>`;
            } else {
                rewardInfoHTML += `<p>Rewards: <span class="no-reward-text">None</span></p>`; // Or just don't show "Rewards:" line
            }
            rewardInfoHTML += `</div>`; // End of .reward-info

            // --- Scherbius Info ---
            let scherbiusInfoHTML = `<div class="scherbius-observed-info">`; // Use this class for styling
            const scherbiusPlayForBattle = (scherbiusObservedPlays && scherbiusObservedPlays[i]) ? scherbiusObservedPlays[i] : [];
            
            let scherbiusCardsContent = '<div class="display-card-container">';
            if (scherbiusPlayForBattle.length > 0) {
                scherbiusPlayForBattle.forEach(cardVal => {
                    scherbiusCardsContent += `<div class="display-card scherbius-card">${cardVal}</div>`;
                });
            } else {
                scherbiusCardsContent += '<span class="no-play-text">None</span>';
            }
            scherbiusCardsContent += '</div>';
            scherbiusInfoHTML += `<p>Scherbius Played: ${scherbiusCardsContent}</p>`;
            scherbiusInfoHTML += `</div>`; // End of .scherbius-observed-info


            // Card Slots Area (same as before)
            const turingPlayedCardsArea = document.createElement('div');
            turingPlayedCardsArea.classList.add('turing-played-cards-area');
            turingPlayedCardsArea.dataset.battleId = battleId;

            for (let slotIdx = 0; slotIdx < clientState.maxCardsPerBattle; slotIdx++) {
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

    // --- Initial Setup (Same) ---
    gameAreaEl.style.display = 'none';
    gameOverMessageEl.style.display = 'none';
    lastRoundSummaryAreaEl.style.display = 'none';
});