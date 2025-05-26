document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements (same as before) ---
    const turingScoreEl = document.getElementById('turingScore');
    const scherbiusScoreEl = document.getElementById('scherbiusScore');
    const maxVictoryPointsEl = document.getElementById('maxVictoryPoints');
    const scherbiusDidEncryptGlobalEl = document.getElementById('scherbiusDidEncryptGlobal');
    const winnerEl = document.getElementById('winner');
    const newGameBtn = document.getElementById('newGameBtn');
    const submitTuringActionBtn = document.getElementById('submitTuringActionBtn');
    const gameAreaEl = document.getElementById('gameArea');
    const gameOverMessageEl = document.getElementById('gameOverMessage');
    const lastRoundSummaryAreaEl = document.getElementById('lastRoundSummaryArea');
    const lastRoundEncryptionInfoEl = document.getElementById('lastRoundEncryptionInfo');
    const lastRoundSummaryBattlesEl = document.getElementById('lastRoundSummaryBattles');
    const rewardsDisplayEl = document.getElementById('rewardsDisplay');
    const turingHandEl = document.getElementById('turingHand');
    const encryptionGuessSlotsEl = document.getElementById('encryptionGuessSlots');
    const encryptionCodeLenEl = document.getElementById('encryptionCodeLen');

    // --- Client-Side Game State ---
    let clientState = {
        initialHandForTurn: [], // Array of {id: "tcard_X", value: Y}
        currentHandDisplayObjects: [], // Cards currently visually in the hand area {id, value}
        battlePlays: {},        // { "battle_0": [{id, value}, {id, value}], ... }
        encryptionGuess: [],    // Array of {id, value} or null, length of encCodeLen
        nBattles: 0, encCodeLen: 0,
        draggedCard: { id: null, value: null, originType: null, originId: null, element: null }
    };

    // --- API Calls (same as before) ---
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

    // --- Main UI Update Function ---
    function updateGlobalUI(serverState) {
        console.log("Server state received:", serverState);
        gameAreaEl.style.display = 'block';

        turingScoreEl.textContent = serverState.turing_points;
        scherbiusScoreEl.textContent = serverState.scherbius_points;
        maxVictoryPointsEl.textContent = serverState.max_victory_points;
        scherbiusDidEncryptGlobalEl.textContent = serverState.scherbius_did_encrypt ? 'Yes' : 'No';
        
        clientState.nBattles = serverState.n_battles;
        clientState.encCodeLen = serverState.encryption_code_len;
        encryptionCodeLenEl.textContent = clientState.encCodeLen;

        if (serverState.current_phase === "Turing_Action") {
            // serverState.turing_hand is now array of {id, value}
            clientState.initialHandForTurn = [...(serverState.turing_hand || [])]; 
            clientState.battlePlays = {};
            for(let i=0; i < clientState.nBattles; i++) clientState.battlePlays[`battle_${i}`] = [];
            clientState.encryptionGuess = new Array(clientState.encCodeLen).fill(null);
        }
        
        renderAllPlayableCardAreas();
        renderRewardsAndBattleDropZones(serverState.rewards, serverState.n_battles, serverState.scherbius_observed_plays);
        
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
    function createCardElement(cardObject, originType, originId) { // cardObject is {id, value}
        const cardDiv = document.createElement('div');
        cardDiv.classList.add('card');
        cardDiv.textContent = cardObject.value; // Display value
        cardDiv.draggable = true;
        cardDiv.dataset.cardId = cardObject.id;     // Store unique ID
        cardDiv.dataset.cardValue = cardObject.value; // Store value for convenience if needed
        cardDiv.dataset.originType = originType;
        cardDiv.dataset.originId = originId;

        cardDiv.addEventListener('dragstart', (e) => {
            clientState.draggedCard = { 
                id: e.target.dataset.cardId, // Use unique ID
                value: parseInt(e.target.dataset.cardValue), 
                originType: e.target.dataset.originType,
                originId: e.target.dataset.originId,
                element: e.target
            };
            e.dataTransfer.setData('text/plain', e.target.dataset.cardId); // Transfer ID
            setTimeout(() => e.target.classList.add('dragging'), 0);
        });
        cardDiv.addEventListener('dragend', (e) => {
            e.target.classList.remove('dragging');
            clientState.draggedCard = { id: null, value: null, originType: null, originId: null, element: null };
        });
        return cardDiv;
    }

    function findCardInArrayById(cardArray, cardId) {
        return cardArray.find(card => card && card.id === cardId);
    }
    function removeCardFromArrayById(cardArray, cardId) {
        const index = cardArray.findIndex(card => card && card.id === cardId);
        if (index > -1) {
            return cardArray.splice(index, 1)[0]; // Remove and return the card object
        }
        return null;
    }

    function renderAllPlayableCardAreas() {
        let cardsEffectivelyInHandObjects = [...clientState.initialHandForTurn];
        
        // Filter out cards in battlePlays by ID
        Object.values(clientState.battlePlays).flat().forEach(playedCardObj => {
            if (playedCardObj) { // Ensure playedCardObj is not null/undefined
                cardsEffectivelyInHandObjects = cardsEffectivelyInHandObjects.filter(handCardObj => handCardObj.id !== playedCardObj.id);
            }
        });
        // Filter out cards in encryptionGuess by ID
        clientState.encryptionGuess.forEach(guessedCardObj => {
            if (guessedCardObj) { // Ensure guessedCardObj is not null/undefined
                cardsEffectivelyInHandObjects = cardsEffectivelyInHandObjects.filter(handCardObj => handCardObj.id !== guessedCardObj.id);
            }
        });
        clientState.currentHandDisplayObjects = cardsEffectivelyInHandObjects;

        turingHandEl.innerHTML = '';
        clientState.currentHandDisplayObjects.forEach(cardObj => {
            turingHandEl.appendChild(createCardElement(cardObj, 'hand', 'turingHand'));
        });

        for (let i = 0; i < clientState.nBattles; i++) {
            const battleId = `battle_${i}`;
            const cardsArea = rewardsDisplayEl.querySelector(`.turing-played-cards-area[data-battle-id="${battleId}"]`);
            if (cardsArea) {
                cardsArea.innerHTML = '';
                (clientState.battlePlays[battleId] || []).forEach(cardObj => {
                    if(cardObj) cardsArea.appendChild(createCardElement(cardObj, 'battle', battleId));
                });
            }
        }
        encryptionGuessSlotsEl.innerHTML = '';
        clientState.encryptionGuess.forEach((cardObj, index) => {
            const slotContainer = document.createElement('div');
            slotContainer.classList.add('encryption-guess-slot');
            slotContainer.dataset.guessSlotId = `guess_${index}`;
            addDropListeners(slotContainer, 'guess', `guess_${index}`);
            if (cardObj) { // cardObj is {id, value} or null
                slotContainer.appendChild(createCardElement(cardObj, 'guess', `guess_${index}`));
            }
            encryptionGuessSlotsEl.appendChild(slotContainer);
        });
    }
    
    function handleCardDrop(targetType, targetId) {
        const { id: draggedCardId, value: draggedCardValue, originType, originId } = clientState.draggedCard;
        if (!draggedCardId) return;

        let cardObjectToMove = null; 

        // 1. Remove card object from its logical origin and get the object
        if (originType === 'hand') {
            // Find the card object from initialHandForTurn that matches the dragged ID
            cardObjectToMove = findCardInArrayById(clientState.initialHandForTurn, draggedCardId);
        } else if (originType === 'battle') {
            const battleCards = clientState.battlePlays[originId];
            cardObjectToMove = removeCardFromArrayById(battleCards, draggedCardId);
        } else if (originType === 'guess') {
            const guessIndex = parseInt(originId.split('_')[1]);
            if (clientState.encryptionGuess[guessIndex] && clientState.encryptionGuess[guessIndex].id === draggedCardId) {
                cardObjectToMove = clientState.encryptionGuess[guessIndex];
                clientState.encryptionGuess[guessIndex] = null;
            }
        }

        if (!cardObjectToMove) {
            console.error("Dragged card object not found in origin!", clientState.draggedCard);
            renderAllPlayableCardAreas(); // Re-render to correct any visual glitch
            return;
        }

        // 2. Add card object to its logical destination
        if (targetType === 'hand') {
            // No explicit add needed; removal from other areas + re-render handles it
        } else if (targetType === 'battle') {
            if (!clientState.battlePlays[targetId]) clientState.battlePlays[targetId] = [];
            // Prevent adding same card instance multiple times
            if (!findCardInArrayById(clientState.battlePlays[targetId], cardObjectToMove.id)) {
                 clientState.battlePlays[targetId].push(cardObjectToMove);
            }
        } else if (targetType === 'guess') {
            const guessIndex = parseInt(targetId.split('_')[1]);
            const oldCardInSlot = clientState.encryptionGuess[guessIndex];
            // If a different card is already in the slot, it's implicitly returned to hand pool
            // by not being in battlePlays or encryptionGuess anymore after this assignment.
            clientState.encryptionGuess[guessIndex] = cardObjectToMove;
        }
        
        renderAllPlayableCardAreas();
    }

    function addDropListeners(element, targetType, targetId) {
        element.addEventListener('dragover', e => { e.preventDefault(); element.classList.add('drag-over'); });
        element.addEventListener('dragleave', e => { element.classList.remove('drag-over'); });
        element.addEventListener('drop', e => {
            e.preventDefault();
            element.classList.remove('drag-over');
            handleCardDrop(targetType, targetId);
        });
    }
    addDropListeners(turingHandEl, 'hand', 'turingHand');

    // --- UI Rendering for Specific Sections ---
    function renderRewardsAndBattleDropZones(rewardsData, numBattles, scherbiusObservedPlays) {
        rewardsDisplayEl.innerHTML = '';
        for (let i = 0; i < numBattles; i++) {
            const battleId = `battle_${i}`;
            const battleDiv = document.createElement('div');
            battleDiv.classList.add('battle-item');
            
            let rewardInfoHTML = `<div class="reward-info"><h4>Battle ${i}</h4>`;
            const vpReward = rewardsData.vp_rewards[i];
            const cardRewardArray = rewardsData.card_rewards[i];
            if (vpReward > 0) rewardInfoHTML += `<p>VP Reward: ${vpReward}</p>`;
            if (cardRewardArray && cardRewardArray.length > 0) rewardInfoHTML += `<p>Card Rewards: ${cardRewardArray.join(', ')}</p>`;
            if (vpReward === 0 && (!cardRewardArray || cardRewardArray.length === 0)) rewardInfoHTML += `<p>No direct reward</p>`;
            rewardInfoHTML += `</div>`;

            let scherbiusInfoHTML = `<div class="scherbius-observed-info">`;
            const scherbiusPlayForBattle = (scherbiusObservedPlays && scherbiusObservedPlays[i]) ? scherbiusObservedPlays[i] : [];
            if (scherbiusPlayForBattle.length > 0) {
                // Display actual (potentially encrypted) card values
                scherbiusInfoHTML += `<p>Scherbius played: ${scherbiusPlayForBattle.join(', ')}</p>`;
            } else {
                scherbiusInfoHTML += `<p>Scherbius played: None</p>`;
            }
            scherbiusInfoHTML += `</div>`;

            const dropAreaHTML = `<div class="turing-played-cards-area" data-battle-id="${battleId}"></div>`;
            battleDiv.innerHTML = rewardInfoHTML + scherbiusInfoHTML + dropAreaHTML;
            rewardsDisplayEl.appendChild(battleDiv);

            const dropArea = battleDiv.querySelector('.turing-played-cards-area');
            addDropListeners(dropArea, 'battle', battleId);
        }
    }
    
    function renderLastRoundSummary(summary) {
        lastRoundEncryptionInfoEl.innerHTML = `Encryption last round: 
            Attempted by Turing: ${summary.encryption_attempted_by_turing ? 'Yes' : 'No'}. 
            Scherbius Encrypted: ${summary.scherbius_encrypted_last_round ? 'Yes' : 'No'}.
            Result: ${summary.encryption_broken_this_round ? '<strong>Broken!</strong>' : 'Not Broken'}.`;

        lastRoundSummaryBattlesEl.innerHTML = '';
        summary.battle_details.forEach(battle => {
            const reportDiv = document.createElement('div');
            reportDiv.classList.add('summary-battle-report');
            reportDiv.innerHTML = `
                <h5>Battle ${battle.battle_id}</h5>
                <p>Turing Played: <span class="cards-list">${battle.turing_played.length > 0 ? battle.turing_played.join(', ') : 'None'}</span></p>
                <p>Scherbius Committed: <span class="cards-list">${battle.scherbius_committed.length > 0 ? battle.scherbius_committed.join(', ') : 'None'}</span></p>
            `; // Assuming scherbius_committed in summary is also array of values
            lastRoundSummaryBattlesEl.appendChild(reportDiv);
        });
    }

    // --- Event Listeners ---
    newGameBtn.addEventListener('click', () => {
        lastRoundSummaryAreaEl.style.display = 'none';
        gameOverMessageEl.style.display = 'none';
        fetchApi('/new_game', 'POST');
    });

    submitTuringActionBtn.addEventListener('click', () => {
        const finalTuringStrategyValues = [];
        for (let i = 0; i < clientState.nBattles; i++) {
            // Map card objects in battlePlays back to just their values
            finalTuringStrategyValues.push(
                (clientState.battlePlays[`battle_${i}`] || []).map(cardObj => cardObj.value)
            );
        }
        
        const finalTuringGuessesValues = (clientState.encryptionGuess.filter(obj => obj !== null).length === clientState.encCodeLen && clientState.encCodeLen > 0) ?
                                   // Map card objects in encryptionGuess back to just their values
                                   [clientState.encryptionGuess.filter(obj => obj !== null).map(cardObj => cardObj.value)] 
                                   : [];

        const payload = { 
            turing_strategy: finalTuringStrategyValues, 
            turing_guesses: finalTuringGuessesValues 
        };
        console.log("Submitting to backend (values):", payload);
        fetchApi('/submit_turing_action', 'POST', payload);
    });

    // --- Initial Setup ---
    gameAreaEl.style.display = 'none';
    gameOverMessageEl.style.display = 'none';
    lastRoundSummaryAreaEl.style.display = 'none';
});