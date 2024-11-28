
# sherbius_player = random_sherbius_player

# @app.route("/")
# def index():
#     if 'game_state' not in session:
#         with open('config.json', 'r') as f:
#             config = json.loads(f.read())
#         config = PyGameConfig(**config)
#         game_state = PyGameState(config)
#         session['game_state'] = game_state.to_dict()
#     else:
#         game_state = PyGameState.from_dict(session['game_state'])

#     return render_template("index.html", **game_state.to_dict())

# @app.route("/action", methods=["POST"])
# def action():
#     game_state = PyGameState.from_dict(session['game_state'])

#     scherbius_strategy, reencrypt = sherbius_player()

#     # fetch turing_strategy from the form
#     turing_strategy = int(request.form['turing_strategy'])
#     turing_guesses = request.form.getlist('turing_guesses')

#     # validate the action
#     if not validate_turing_action(turing_strategy, turing_guesses):
#         pass # not sure how to handle this yet

#     game_state.step(
#         turing_strategy, 
#         scherbius_strategy,
#         turing_guesses, 
#         reencrypt
#     )

#     session['game_state'] = game_state.to_dict()

#     return redirect(url_for('index'))