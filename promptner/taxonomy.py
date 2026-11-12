"""FewNERD fine-grained entity labels and concise English definitions."""

TYPE_DEFINITIONS = {
    "art-broadcastprogram": "A named television or radio program, series, or broadcast show.",
    "art-film": "A named motion picture or film, including a specific movie.",
    "art-music": "A named song, album, musical composition, or other music work.",
    "art-other": "A named creative work that is not a film, music, broadcast program, painting, or written work.",
    "art-painting": "A named painting or other individually identified visual artwork of that kind.",
    "art-writtenart": "A named written work, such as a book, poem, play, article, or literary work.",
    "building-airport": "A named airport or airfield facility.",
    "building-hospital": "A named hospital or other facility primarily providing medical care.",
    "building-hotel": "A named hotel or lodging establishment.",
    "building-library": "A named library or major named collection facility.",
    "building-other": "A named constructed building or facility not covered by another building subtype.",
    "building-restaurant": "A named restaurant or food-service establishment.",
    "building-sportsfacility": "A named sports venue or facility used for athletic events or training.",
    "building-theater": "A named theater or performance venue.",
    "event-attack/battle/war/militaryconflict": "A named attack, battle, war, or other military conflict.",
    "event-disaster": "A named natural or human-caused disaster or catastrophic incident.",
    "event-election": "A named election or electoral contest.",
    "event-other": "A named event not covered by another event subtype.",
    "event-protest": "A named protest, demonstration, or organized public resistance event.",
    "event-sportsevent": "A named sports match, tournament, championship, or athletic competition.",
    "location-GPE": "A named geopolitical entity, such as a country, state, city, or other political territory.",
    "location-bodiesofwater": "A named ocean, sea, river, lake, or other body of water.",
    "location-island": "A named island or island group.",
    "location-mountain": "A named mountain, mountain range, or major peak.",
    "location-other": "A named geographic place not covered by another location subtype.",
    "location-park": "A named park or protected recreational or natural area.",
    "location-road/railway/highway/transit": "A named road, railway, highway, transit route, or transport line.",
    "organization-company": "A named commercial company or business.",
    "organization-education": "A named school, university, or other educational institution.",
    "organization-government/governmentagency": "A named government, ministry, department, or government agency.",
    "organization-media/newspaper": "A named newspaper, news outlet, or media organization.",
    "organization-other": "A named organization not covered by another organization subtype.",
    "organization-politicalparty": "A named political party or political organization.",
    "organization-religion": "A named religious organization, denomination, or institution.",
    "organization-showorganization": "A named organization that produces, stages, or organizes entertainment shows.",
    "organization-sportsleague": "A named sports league or governing competition organization.",
    "organization-sportsteam": "A named sports club or team.",
    "other-astronomything": "A named astronomical object or phenomenon, such as a star, planet, or galaxy.",
    "other-award": "A named prize, award, medal, or honor.",
    "other-biologything": "A named biological entity or concept, such as a species, organism, or biological structure.",
    "other-chemicalthing": "A named chemical substance, compound, or chemical concept.",
    "other-currency": "A named currency or monetary unit.",
    "other-disease": "A named disease, disorder, or pathological condition.",
    "other-educationaldegree": "A named academic degree or educational qualification.",
    "other-god": "A named deity or god from a religion or mythology.",
    "other-language": "A named natural or constructed language.",
    "other-law": "A named law, act, bill, statute, or legal instrument.",
    "other-livingthing": "A named living organism or biological individual not covered by a narrower type.",
    "other-medical": "A named medical treatment, procedure, medication, or medical concept other than a disease.",
    "person-actor": "A person identified primarily as an actor or performer in film, television, or theater.",
    "person-artist/author": "A person identified primarily as an artist, author, writer, or creator of artistic works.",
    "person-athlete": "A person identified primarily as an athlete or sportsperson.",
    "person-director": "A person identified primarily as a film, television, or theater director.",
    "person-other": "A named person not covered by another person subtype.",
    "person-politician": "A person identified primarily as a politician or elected political official.",
    "person-scholar": "A person identified primarily as an academic, researcher, or scholar.",
    "person-soldier": "A person identified primarily as a soldier or military service member.",
    "product-airplane": "A named airplane, aircraft model, or other aircraft product.",
    "product-car": "A named car, automobile model, or other road vehicle product.",
    "product-food": "A named food, dish, or branded food product.",
    "product-game": "A named game, including a video game, board game, or sports game product.",
    "product-other": "A named product not covered by another product subtype.",
    "product-ship": "A named ship, boat, or vessel product.",
    "product-software": "A named software application, operating system, or software product.",
    "product-train": "A named train, locomotive, or rail vehicle product.",
    "product-weapon": "A named weapon or weapon system product.",
}

ENTITY_TYPES = tuple(TYPE_DEFINITIONS)

COARSE_DEFINITIONS = {
    "art": "A named creative work, including a film, book, song, painting, or broadcast program.",
    "building": "A named constructed facility or venue, such as a school, hospital, airport, or stadium.",
    "event": "A named event, such as an election, protest, disaster, battle, or sports competition.",
    "location": "A named geographic or geopolitical place, including countries, cities, rivers, and roads.",
    "organization": "A named organized group or institution, such as a company, government, school, team, or party.",
    "other": "A named concept or thing such as a disease, award, language, law, species, or chemical.",
    "person": "A named individual person.",
    "product": "A named manufactured or commercial product, such as a car, software, ship, or game.",
}

COARSE_TYPES = tuple(COARSE_DEFINITIONS)
COARSE_TO_FINE = {
    coarse: tuple(label for label in ENTITY_TYPES if label.split("-", 1)[0] == coarse)
    for coarse in COARSE_TYPES
}


def coarse_type(fine_type: str) -> str:
    """Return the FewNERD superclass of an exact fine-grained type."""
    coarse, separator, _ = fine_type.partition("-")
    if not separator or coarse not in COARSE_DEFINITIONS:
        raise ValueError(f"Unknown fine-grained entity type: {fine_type}")
    return coarse

